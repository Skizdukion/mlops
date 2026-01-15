import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlmodel import Session, create_engine
from evidently import Report
from evidently.presets import DataDriftPreset

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from monitoring.models import (
    DatadriftsMetrics,
)

from api_gateway.app.config import config

# Setup Database Engine
engine = create_engine(config.DATABASE_URL)

LOOP_INTERVAL_SECONDS = 60  # Run every minute
CURRENT_DATA_LIMIT = 500  # Last X rows as current data
REFERENCE_DATA_SAMPLE_SIZE = 50000  # Sample size for huge training data


def load_reference_data(model_type: str) -> pd.DataFrame:
    """Load training data marked as current reference, using sampling."""
    # Using random sampling to avoid loading millions of rows
    query = f"""
        SELECT 
            p.tpep_pickup_datetime, p.predicted_duration as prediction, f.duration as target
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = '{model_type}'
          AND f.is_current_train = True
        ORDER BY RANDOM()
        LIMIT {REFERENCE_DATA_SAMPLE_SIZE}
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error loading reference data for {model_type}: {e}")
        return pd.DataFrame()


def load_current_data(model_type: str, limit=500) -> pd.DataFrame:
    """Fetch recent data for drift detection."""
    query = f"""
        SELECT 
            p.tpep_pickup_datetime, p.predicted_duration as prediction, f.duration as target
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = '{model_type}'
        ORDER BY p.tpep_pickup_datetime DESC
        LIMIT {limit}
    """
    try:
        return pd.read_sql(query, engine)
    except Exception as e:
        print(f"Error loading current data for {model_type}: {e}")
        return pd.DataFrame()


def calculate_drift(reference_df, current_df):
    if reference_df.empty or current_df.empty:
        return None

    # Evidently Drift Calculation
    # Note: In 0.7.20 Report(metrics=[DataDriftPreset()]) works
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    return report.dict()


def save_drift_metrics(drift_data, model_type):
    if not drift_data:
        return

    try:
        # Extract metrics from Evidently Report
        # Structure usually: metrics -> [ { metric: "DataDriftTable", result: { ... } } ]
        metrics = drift_data["metrics"][0]["result"]

        # evidently 0.x usually provides these
        dataset_drift = metrics.get("dataset_drift", False)
        drift_share = metrics.get("drift_share", 0.0)
        # number_of_drifted_columns = metrics.get("number_of_drifted_columns", 0)

        drifted_columns = {}
        # If needed, can iterate metrics['drift_by_columns'] for details
        # For now, just storing empty dict or could parse if requirements change

        metric_record = DatadriftsMetrics(
            timestamp=datetime.utcnow(),
            model_type=model_type,
            data_drift=drift_share,
            drifted_columns=drifted_columns,
        )

        with Session(engine) as session:
            session.add(metric_record)
            session.commit()
            print(
                f"[{model_type}] Drift saved: Share={drift_share:.4f}, Drifted={dataset_drift}"
            )

    except Exception as e:
        print(f"Error saving drift for {model_type}: {e}")


def main():
    print("Data Drift Worker started (Evidently with Sampling)...")

    while True:
        active_models = ["catboost", "xgboost"]
        # active_models = ["xgboost"]

        for model in active_models:
            print(f"Checking drift for {model}...")

            ref_df = load_reference_data(model)
            curr_df = load_current_data(model, CURRENT_DATA_LIMIT)

            if not ref_df.empty and not curr_df.empty:
                print(f"  Ref Size: {len(ref_df)}, Curr Size: {len(curr_df)}")
                drift_result = calculate_drift(ref_df, curr_df)
                if drift_result:
                    save_drift_metrics(drift_result, model)
            else:
                print(f"  Skipping {model}: Ref={len(ref_df)}, Curr={len(curr_df)}")

        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
