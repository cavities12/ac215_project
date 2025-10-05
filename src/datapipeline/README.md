# Data Pipeline
> [!NOTE]
> This module is designed to preprocess and load datasets for machine learning tasks.
> It handles the combination of multiple CSV files, data cleaning, dataset creation, and splitting the data into training, validation, and test sets.
> It also supports uploading datasets to Google Cloud Platform (GCP).

## Storage Layout

TODO: put the tree here

## Preprocessing

TODO: downloading, converting -> `.parquet`, cleaning...

## Instructions
1. Start a container with:

```shell
./docker-shell.sh
```

2. Once the container is started, the preprocessing pipeline can be started with:

```shell
python preprocess.py
```
