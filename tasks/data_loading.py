import pandas as pd
from urllib.parse import urlparse
import os
import requests
from typing import Union

def nyc_data_loading(parquet_urls: Union[str, list[str]], cache_dir="data/nyc-taxi"):
    if isinstance(parquet_urls, str):
        parquet_urls = [parquet_urls]

    # Ensure the local directory exists
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Read and concatenate all Parquet files
    df_list = []
    for url in parquet_urls:
        # Extract filename from URL to create a local path
        filename = os.path.basename(urlparse(url).path)
        local_path = os.path.join(cache_dir, filename)

        # Step 1: Check if file exists locally; if not, download it
        if not os.path.exists(local_path):
            print(f"Downloading: {url}...")
            response = requests.get(url)
            with open(local_path, "wb") as f:
                f.write(response.content)
        else:
            print(f"Loading from cache: {local_path}")

        # Step 2: Read from the local path
        df_part = pd.read_parquet(local_path)

        # Standardize columns
        df_part.columns = df_part.columns.str.lower().str.replace(" ", "_")
        df_list.append(df_part)

    # Combine all dataframes
    df = pd.concat(df_list, ignore_index=True)

    return df
