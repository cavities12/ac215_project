import io
import os
import pandas as pd

from google.cloud import storage
from pathlib import Path

# GCS bucket name.
BUCKET = "accimap-data"

# GCP service account key.
# TODO: move this into Dockerfile or docker-shell.sh
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../../secrets/data-sa.json"


def main():
    client = storage.Client()
    bucket = client.bucket(BUCKET)

    for blob in client.list_blobs(BUCKET, prefix="bronze/csv/"):
        if not blob.name.endswith(".csv"):
            continue

        # TODO: better logging

        # Create a polars dataframe from the csv file.
        print(f"Downloading {blob.name}...")
        with blob.open("rb") as f:
            df = pd.read_csv(f)
        print(f"Downloaded {blob.name}!")

        # Write into a local buffer.
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)

        # Rewrite to a parquet file.
        dst = Path("bronze/parquet") / Path(blob.name).relative_to("bronze/csv")
        dst = dst.with_suffix(".parquet")

        # Upload the parquet file.
        print(f"Uploading {str(dst)}...")
        bucket.blob(str(dst)).upload_from_file(buf, rewind=True)
        print(f"Uploaded {str(dst)}!")


if __name__ == "__main__":
    main()
