import sys
from pathlib import Path

import pandas as pd
import uuid
from datetime import datetime
from sqlalchemy import create_engine, insert
from sqlmodel import Session


# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# from prefect import task, get_run_logger
from alembic_model.config import config
from alembic_model.models.domain import Prediction, Feedback

# DB Config
engine = create_engine(config.DATABASE_URL)


def populate_initial_training_data(
    df: pd.DataFrame, model_type: str, batch_size: int = 100000
):
    """
    Populates the database with the initial training data for drift monitoring using bulk insert.
    Marks these records as is_current_train=True.
    """
    now = datetime.utcnow()

    # Limit initial data to 500k rows
    MAX_ROWS = 500000
    if len(df) > MAX_ROWS:
        print(
            f"[{model_type}] Input data size {len(df)} exceeds limit {MAX_ROWS}. Sampling down..."
        )
        df = df.sample(n=MAX_ROWS, random_state=42)

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
    col_tpep = get_col("tpep_pickup_datetime")
    col_tdep = get_col("tpep_dropoff_datetime")
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

            # To be safe against "PULocationID" vs "pulocationid" in named tuple attribute access:
            # Pandas renames invalid identifiers.
            # Let's assume standard behavior: getattr(row, col_name)

            def safe_int(val, default=0):
                try:
                    if pd.isna(val):
                        return default
                    return int(val)
                except (ValueError, TypeError):
                    return default

            try:
                # Extract values
                tpep = getattr(row, col_tpep)
                tdep = getattr(row, col_tdep)
                pass_count = safe_int(getattr(row, col_pass))
                pu_loc = safe_int(getattr(row, col_pu))
                do_loc = safe_int(getattr(row, col_do))

                # Calculate duration in minutes (since it's not in raw data yet)
                duration = (tdep - tpep).total_seconds() / 60.0
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
