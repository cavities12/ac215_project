import io
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from google.cloud import storage
from pathlib import Path
from tempfile import TemporaryDirectory

# GCS bucket name.
BUCKET = "accimap-data"

# Streaming chunk size.
CHUNK_SIZE = 100_000


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a dataframe."""
    # TODO: what columns do we want?
    return df


def main():
    """Entry point for all preprocessing tasks."""
    client = storage.Client()
    bucket = client.bucket(BUCKET)
    print("Client successfully established connection with GCP.")

    for blob in client.list_blobs(BUCKET, prefix="bronze/csv/"):
        if not blob.name.endswith(".csv"):
            continue

        # Rewrite to a parquet file.
        dst = Path("bronze/parquet") / Path(blob.name).relative_to("bronze/csv")
        dst = dst.with_suffix(".parquet")

        # Temporary local file for dataframe conversion.
        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / dst.name
            writer = None

            # Read the .csv file in chunks of 100,000 rows.
            with blob.open("rb") as f:
                for chunk in pd.read_csv(f, chunksize=CHUNK_SIZE):
                    chunk = clean(chunk)
                    table = pa.Table.from_pandas(chunk, preserve_index=False)
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, table.schema)

                    writer.write_table(table)
                    print("Wrote chunk...")

            if writer is None:
                print(f"No rows were written for {blob.name}, skipping...")
                continue

            print(f"Finished downloading {blob.name}.")
            writer.close()
            bucket.blob(str(dst)).upload_from_filename(str(tmp_path))
            print(f"Finished uploading {blob.name}.")


if __name__ == "__main__":
    main()
