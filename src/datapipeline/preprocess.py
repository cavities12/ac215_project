import io
import os
import sys
import warnings
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import requests
import h3
import pytz
import holidays
from datetime import datetime, timezone
from typing import List, Dict, Tuple, Optional

from google.cloud import storage
from pathlib import Path
from tempfile import TemporaryDirectory

# Add relative import path
sys.path.append(str(Path(__file__).parent.parent))

# GCS bucket name.
BUCKET = "accimap-data"

# Streaming chunk size.
CHUNK_SIZE = 200_000

# Configuration constants from notebook
BBOX = {
    "min_lat": 33.5,
    "min_lng": -119.0,
    "max_lat": 34.9,
    "max_lng": -117.0
}

H3_RES = 8
START_UTC = "2022-01-01T00:00:00Z"
END_UTC = "2023-01-01T00:00:00Z"
TRAIN_END = "2022-07-01T00:00:00Z"
VAL_END = "2022-10-01T00:00:00Z"
TEST_END = "2023-01-01T00:00:00Z"

# Required columns
BASE_COLS = ["ID", "Start_Time", "Start_Lat", "Start_Lng", "City", "County", "State", "Timezone"]
INFRA_COLS = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", "Railway",
    "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal", "Turning_Loop"
]

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)


def df_clean(df: pd.DataFrame, csv_path: Optional[str] = None) -> pd.DataFrame:
    """
    Clean and filter accident data according to geographic and temporal bounds.
    
    Args:
        df: Raw accident dataframe
        csv_path: Optional path to CSV for detecting available columns
        
    Returns:
        Cleaned dataframe with filtered accidents
    """
    print(f"Starting with {len(df):,} rows")
    
    # Detect available infrastructure columns
    if csv_path:
        header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
    else:
        header_cols = df.columns.tolist()
    
    available_infra = [c for c in INFRA_COLS if c in header_cols]
    usecols = list(dict.fromkeys(BASE_COLS + available_infra))  # de-dup
    
    # Keep only available columns
    available_cols = [c for c in usecols if c in df.columns]
    df = df[available_cols].copy()
    
    # Geographic filtering (Los Angeles bounding box)
    df = df[
        (df.Start_Lat >= BBOX["min_lat"]) & 
        (df.Start_Lat <= BBOX["max_lat"]) &
        (df.Start_Lng >= BBOX["min_lng"]) & 
        (df.Start_Lng <= BBOX["max_lng"])
    ].copy()
    
    # Convert timestamps
    df["Start_Time"] = pd.to_datetime(df["Start_Time"], utc=True, errors="coerce")
    
    # Remove rows with missing critical data
    df = df.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng"])
    
    # Time window filtering
    start_utc = pd.to_datetime(START_UTC, utc=True)
    end_utc = pd.to_datetime(END_UTC, utc=True)
    df = df[(df["Start_Time"] >= start_utc) & (df["Start_Time"] < end_utc)].copy()
    
    print(f"After filtering: {len(df):,} rows")
    print(f"Infrastructure columns available: {available_infra}")
    
    return df


def create_h3_grid() -> pd.DataFrame:
    """Create H3 hexagonal grid for Los Angeles."""
    # Get LA boundary from OpenStreetMap
    url = "https://nominatim.openstreetmap.org/search.php?q=Los+Angeles+California&polygon_geojson=1&format=json"
    r = requests.get(url, headers={"User-Agent": "LA-grid-demo"}).json()
    geojson_poly = r[0]["geojson"]
    
    # Create H3 polygon
    coords = geojson_poly["coordinates"][0]  # Outer ring
    poly = h3.LatLngPoly([(lat, lng) for lng, lat in coords])
    
    # Generate hex cells
    cells = list(h3.polygon_to_cells(poly, res=H3_RES))
    cells_df = pd.DataFrame({"h3_id": cells})
    
    print(f"H3 res={H3_RES} → {len(cells_df):,} cells inside Los Angeles")
    return cells_df


def create_time_panel(cells_df: pd.DataFrame) -> pd.DataFrame:
    """Create time panel for all cells and hours."""
    start_utc = pd.to_datetime(START_UTC, utc=True)
    end_utc = pd.to_datetime(END_UTC, utc=True)
    
    # Build hours index
    hours = pd.date_range(start=start_utc, end=end_utc, freq="H", inclusive="left", tz="UTC")
    hours_df = pd.DataFrame({"ts_utc": hours})
    
    # Create panel
    cells_df = cells_df.copy()
    cells_df["key"] = 1
    hours_df["key"] = 1
    panel = cells_df.merge(hours_df, on="key").drop(columns=["key"])
    
    panel["ts_utc"] = pd.to_datetime(panel["ts_utc"], utc=True)
    return panel


def map_accidents_to_panel(acc: pd.DataFrame, panel: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Map accidents to panel and create labels."""
    # Compute H3 cells for accidents
    acc = acc.copy()
    acc["h3_id"] = acc.apply(
        lambda r: h3.latlng_to_cell(r["Start_Lat"], r["Start_Lng"], H3_RES), axis=1
    )
    
    # Keep only accidents in LA cells
    la_cells = set(panel["h3_id"].tolist())
    acc = acc[acc["h3_id"].isin(la_cells)].copy()
    
    # Floor times to hour
    acc["ts_utc"] = pd.to_datetime(acc["Start_Time"], utc=True).dt.floor("H")
    
    # Count accidents per cell-hour
    acc_counts = (acc.groupby(["h3_id", "ts_utc"])
                    .size()
                    .rename("cnt")
                    .reset_index())
    
    # Merge with panel
    if "y" in panel.columns:
        panel = panel.drop(columns=["y"])
    
    panel = panel.merge(acc_counts, on=["h3_id", "ts_utc"], how="left")
    panel["y"] = (panel["cnt"].fillna(0) > 0).astype("int8")
    panel = panel.drop(columns=["cnt"])
    
    return panel, acc


def add_static_features(panel: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """Add static infrastructure features."""
    available_infra = [c for c in INFRA_COLS if c in acc.columns]
    print("Infra cols available:", available_infra)
    
    if not available_infra:
        print("No infrastructure columns available; skipping static road features.")
        return panel
    
    def to01(s):
        return s.map({True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}).fillna(0).astype("int8")
    
    for c in available_infra:
        acc[c] = to01(acc[c])
    
    # Aggregate infrastructure features by cell
    static_feats = (
        acc.groupby("h3_id")[available_infra]
           .mean()  # fraction of historical crashes with that attribute
           .reset_index()
    )
    
    panel = panel.merge(static_feats, on="h3_id", how="left")
    for c in available_infra:
        panel[c] = panel[c].fillna(0).astype("float32")
    
    return panel


def add_temporal_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features (hour, day of week, etc.)."""
    ts = pd.to_datetime(panel["ts_utc"], utc=True)
    
    panel["hour"] = ts.dt.hour.astype("int16")
    panel["dow"] = ts.dt.dayofweek.astype("int16")
    panel["month"] = ts.dt.month.astype("int16")
    panel["is_weekend"] = (panel["dow"] >= 5).astype("int8")
    
    # Add holidays
    years = sorted(set(ts.dt.year.tolist()))
    us_holidays = holidays.UnitedStates(years=years)
    panel["is_holiday"] = ts.dt.date.astype("O").map(
        lambda d: 1 if d in us_holidays else 0
    ).astype("int8")
    
    panel = panel.sort_values(["h3_id", "ts_utc"])
    return panel


def compute_lags(panel: pd.DataFrame) -> pd.DataFrame:
    """Compute lag features for each cell."""
    def compute_lags_group(g):
        y = g["y"].astype(int)
        g["lag_1h"] = y.shift(1).fillna(0)
        g["lag_3h"] = y.shift(1).rolling(3).sum().fillna(0)
        g["lag_24h"] = y.shift(1).rolling(24).sum().fillna(0)
        g["lag_7d_sum"] = y.shift(1).rolling(168, min_periods=1).sum()
        g["lag_30d_sum"] = y.shift(1).rolling(720, min_periods=1).sum()
        return g
    
    panel = panel.groupby("h3_id", group_keys=False).apply(compute_lags_group)
    return panel


def create_train_val_test_splits(panel: pd.DataFrame, neg_frac: float = 1.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits with optional negative sampling."""
    train_end = pd.to_datetime(TRAIN_END, utc=True)
    val_end = pd.to_datetime(VAL_END, utc=True)
    test_end = pd.to_datetime(TEST_END, utc=True)
    
    ts = pd.to_datetime(panel["ts_utc"], utc=True)
    
    def split_name(t):
        if t < train_end:
            return "train"
        elif t < val_end:
            return "val"
        elif t < test_end:
            return "test"
        else:
            return "ignore"
    
    panel["split"] = [split_name(t) for t in ts]
    panel = panel[panel["split"] != "ignore"].copy()
    
    # Create splits
    train_df = panel[panel["split"] == "train"].copy()
    val_df = panel[panel["split"] == "val"].copy()
    test_df = panel[panel["split"] == "test"].copy()
    
    # Negative sampling for training
    if neg_frac < 1.0:
        pos = train_df[train_df["y"] == 1]
        neg = train_df[train_df["y"] == 0].sample(frac=neg_frac, random_state=42)
        train_df = pd.concat([pos, neg], ignore_index=True)
    
    # Add sample weights
    for df in (train_df, val_df, test_df):
        df["weight"] = 1.0
    train_df.loc[train_df["y"] == 0, "weight"] = 1.0 / neg_frac
    
    print(f"Train rows: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
    print("Positives in train:", (train_df['y'] == 1).sum())
    
    return train_df, val_df, test_df


def build_feature_pipeline(acc: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Complete feature engineering pipeline.
    
    Args:
        acc: Cleaned accident dataframe
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    # Create H3 grid and time panel
    cells_df = create_h3_grid()
    panel = create_time_panel(cells_df)
    
    # Map accidents to panel
    panel, acc = map_accidents_to_panel(acc, panel)
    
    # Add features
    panel = add_static_features(panel, acc)
    panel = add_temporal_features(panel)
    panel = compute_lags(panel)
    
    # Create splits
    train_df, val_df, test_df = create_train_val_test_splits(panel)
    
    return train_df, val_df, test_df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy clean function for backward compatibility.
    Now uses df_clean internally.
    """
    return df_clean(df)


def process_accident_data_from_gcs():
    """Process accident data from GCS and create training features."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    print("Client successfully established connection with GCP.")

    # Process CSV files and create training features
    for blob in client.list_blobs(BUCKET, prefix="accidents/csv/"):
        if not blob.name.endswith(".csv"):
            continue

        print(f"Processing {blob.name}...")
        
        # Download and process the CSV file
        with TemporaryDirectory() as tmp_dir:
            tmp_csv = Path(tmp_dir) / "raw_data.csv"
            tmp_parquet = Path(tmp_dir) / "processed_data.parquet"
            
            # Download CSV
            with open(tmp_csv, "wb") as f:
                blob.download_to_file(f)
            
            # Read and clean data
            acc_data = pd.read_csv(tmp_csv, low_memory=False)
            acc_clean = df_clean(acc_data)
            
            if len(acc_clean) == 0:
                print(f"No data after cleaning for {blob.name}, skipping...")
                continue
            
            # Build features
            train_df, val_df, test_df = build_feature_pipeline(acc_clean)
            
            # Save processed data as parquet
            processed_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
            processed_data.to_parquet(tmp_parquet, index=False)
            
            # Upload processed data
            dst = Path("accidents/processed") / Path(blob.name).relative_to("accidents/csv")
            dst = dst.with_suffix(".parquet")
            bucket.blob(str(dst)).upload_from_filename(str(tmp_parquet))
            
            print(f"Processed and uploaded features for {blob.name}")
            print(f"  - Train: {len(train_df):,} rows")
            print(f"  - Val: {len(val_df):,} rows") 
            print(f"  - Test: {len(test_df):,} rows")


def main():
    """Entry point for all preprocessing tasks."""
    # Option 1: Process data with full feature engineering
    print("Starting full feature engineering pipeline...")
    process_accident_data_from_gcs()

    # Option 2: Legacy CSV to Parquet conversion (keeping for backward compatibility)
    print("\nAlso running legacy CSV to Parquet conversion...")
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    print("Client successfully established connection with GCP.")

    # 🔧 Predefine a fixed schema so every chunk uses the same column types
    schema = pa.schema([
        ("ID", pa.string()),
        ("Start_Time", pa.timestamp("ns", tz="UTC")),
        ("Start_Lat", pa.float64()),
        ("Start_Lng", pa.float64()),
        ("City", pa.string()),
        ("County", pa.string()),
        ("State", pa.string()),
        ("Timezone", pa.string()),
        *[(col, pa.bool_()) for col in INFRA_COLS]
    ])

    for blob in client.list_blobs(BUCKET, prefix="accidents/csv/"):
        if not blob.name.endswith(".csv"):
            continue

        print(f"Processing {blob.name}...")

        # Destination Parquet path in GCS
        dst = Path("accidents/parquet") / Path(blob.name).relative_to("accidents/csv")
        dst = dst.with_suffix(".parquet")

        # Temporary local file for writing
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / dst.name
            writer = None
            rows_written = 0

            # Stream CSV chunks
            with blob.open("rb") as f:
                for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE, low_memory=False):
                    chunk = clean(chunk)

                    # Skip empty chunks to avoid null-schema issues
                    if chunk.empty:
                        print(" Skipping empty chunk (0 rows).")
                        continue

                    # Convert to Arrow Table with predefined schema
                    table = pa.Table.from_pandas(chunk, schema=schema, preserve_index=False)

                    # Create writer if not yet initialized
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, schema=schema)

                    writer.write_table(table)
                    rows_written += len(chunk)
                    print(f"Wrote chunk with {len(chunk):,} rows...")

            
            if writer:
                writer.close()
                bucket.blob(str(dst)).upload_from_filename(str(tmp_path))
                print(f" Finished uploading {blob.name} → {dst}")
                print(f"   Total rows written: {rows_written:,}")
            else:
                print(f" No valid rows found for {blob.name}, skipped upload.")


if __name__ == "__main__":
    main()
