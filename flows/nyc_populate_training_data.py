import pandas as pd
import uuid
from datetime import datetime
from sqlalchemy import create_engine, insert
from sqlmodel import Session

# from prefect import task, get_run_logger
from api_gateway.app.config import config
from api_gateway.app.models.domain import Prediction, Feedback
from tasks.data_loading import nyc_data_loading
from tasks.feature_engineering.nyc_duration import (
    NycTreeDataFeature,
)

# DB Config
engine = create_engine(config.DATABASE_URL)


def populate_initial_training_data(
    df: pd.DataFrame, model_type: str, batch_size: int = 10000
):
    """
    Populates the database with the initial training data for drift monitoring using bulk insert.
    Marks these records as is_current_train=True.
    """
    now = datetime.utcnow()
    total_rows = len(df)
    print(f"Starting bulk population for {model_type}: {total_rows} rows")

    # Ensure columns derived from DataFrame are consistent
    # Handling potential case differences in column names (e.g., PULocationID vs pulocationid)
    # Mapping: DB Field <- DF Column

    # Create a local copy/view with normalized columns for easier mapping
    # Note: Creating a massive copy might be memory intensive.
    # Better to just check what's available.
    available_cols = set(df.columns)

    # Helper to find column name ignoring case
    def get_col(name):
        if name in available_cols:
            return name
        for c in available_cols:
            if c.lower() == name.lower():
                return c
        return name  # Fallback, might error if missing

    col_pu = get_col("pulocationid")
    col_do = get_col("dolocationid")
    col_dur = get_col("duration")
    col_tpep = get_col("tpep_pickup_datetime")
    col_pass = get_col("passenger_count")

    # Process in batches
    for start_idx in range(0, total_rows, batch_size):
        end_idx = min(start_idx + batch_size, total_rows)
        batch_df = df.iloc[start_idx:end_idx]

        predictions_data = []
        feedbacks_data = []

        for row in batch_df.itertuples(index=False):
            # getattr is safer than row.field if we aren't sure of namedtuple structure
            # but itertuples yields a namedtuple where fields match cleaned column names.
            # However, if column names have spaces or strange chars, itertuples might rename them.
            # reliable way for flexible columns is zip or converting batch to dict,
            # but usually itertuples is fine if columns are standard.
            # Using row access by attribute based on our resolved names.

            # Since we resolved names like 'PULocationID' matches `row.PULocationID` (if valid identifier)
            # We'll trust accessing by attribute mechanism of Pandas named tuples

            # To be safe against "PULocationID" vs "pulocationid" in named tuple attribute access:
            # Pandas renames invalid identifiers.
            # Let's assume standard behavior: getattr(row, col_name)

            try:
                # Extract values
                tpep = getattr(row, col_tpep)
                pass_count = int(getattr(row, col_pass))
                pu_loc = int(getattr(row, col_pu))
                do_loc = int(getattr(row, col_do))
                duration = float(getattr(row, col_dur))
            except AttributeError as e:
                # Fallback implementation if getattr fails (e.g. index/naming issues)
                # Slower per row but checks might save full crash
                print(f"Error accessing row attributes: {e}. Check DataFrame columns.")
                return

            pred_id = uuid.uuid4()
            feedback_id = uuid.uuid4()

            predictions_data.append(
                {
                    "id": pred_id,
                    "tpep_pickup_datetime": tpep,
                    "passenger_count": pass_count,
                    "pulocationid": pu_loc,
                    "dolocationid": do_loc,
                    "model_type": model_type,
                    "predicted_duration": None,
                    "created_at": now,
                }
            )

            feedbacks_data.append(
                {
                    "id": feedback_id,
                    "prediction_id": pred_id,
                    "dolocationid": do_loc,
                    "duration": duration,
                    "is_current_train": True,
                    "created_at": now,
                }
            )

        # Bulk Insert for the batch
        with Session(engine) as session:
            session.execute(insert(Prediction), predictions_data)
            session.execute(insert(Feedback), feedbacks_data)
            session.commit()

        print(f"Inserted batch {start_idx}-{end_idx}")

    print("DB Population complete.")


if __name__ == "__main__":
    model_types = ["xgboost", "rf", "elastic"]

    train_urls = [
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet",
    ]

    train_df = nyc_data_loading(train_urls)

    # Note: If NycTreeDataFeature transforms/removes columns (e.g. OHE), this might lack PULocationID
    # Ideally pass 'train_df' directly if raw columns are needed.
    feature_engineer = NycTreeDataFeature()
    df_processed = feature_engineer.fit_transform(train_df)

    # Using train_df (raw) which definitely has the ID columns
    for model_type in model_types:
        populate_initial_training_data(train_df, model_type)
