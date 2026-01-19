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

from alembic_model.config import alembic_config

# Setup Database Engine
engine = create_engine(alembic_config.DATABASE_URL)

LOOP_INTERVAL_SECONDS = 60  # Run every minute
CURRENT_DATA_LIMIT = 1000  # Last X rows as current data
REFERENCE_DATA_SAMPLE_SIZE = 50000  # Sample size for huge training data


def load_reference_data(model_type: str) -> pd.DataFrame:
    """Load training data marked as current reference, using sampling."""
    # Using random sampling to avoid loading millions of rows
    query = f"""
        SELECT 
            p.tpep_pickup_datetime, 
            p.predicted_duration as prediction, 
            f.duration as target,
            p.passenger_count,
            p.pulocationid,
            p.dolocationid
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
            p.tpep_pickup_datetime, 
            p.predicted_duration as prediction, 
            f.duration as target,
            p.passenger_count,
            p.pulocationid,
            p.dolocationid
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
    # report.run returns the calculated report (snapshot in some versions)
    results = report.run(reference_data=reference_df, current_data=current_df)
    return results.dict()


def save_drift_metrics(drift_data, model_type):
    if not drift_data:
        return

    try:
        # Extract metrics from Evidently Report
        # We need to find the metric that contains 'drift_share'
        metrics_list = drift_data.get("metrics", [])
        dataset_drift = False
        drift_share = 0.0

        found = False
        for metric_item in metrics_list:
            # Check for DriftedColumnsCount metric structure
            # Example: {"metric_name": "DriftedColumnsCount...", "config": { "type": "evidently:metric_v2:DriftedColumnsCount", "drift_share": 0.5 }, "value": {"count": 0.0, "share": 0.0}}

            config = metric_item.get("config", {})
            metric_type = config.get("type", "")

            if (
                "DriftedColumnsCount" in metric_type
                or "DriftedColumnsCount" in metric_item.get("metric_name", "")
            ):
                val = metric_item.get("value", {})
                if isinstance(val, dict):
                    drift_share = val.get("share", 0.0)
                    threshold = config.get("drift_share", 0.5)
                    # Boolean: is share >= threshold?
                    dataset_drift = drift_share >= threshold
                    found = True
                    break

        if not found:
            # Fallback debug or warning
            first_item = metrics_list[0] if metrics_list else "Empty"
            print(
                f"[{model_type}] Warning: Could not find drift metrics. First item string: {str(first_item)[:100]}"
            )

        # Extract per-column drift status
        drifted_columns = {}
        for metric_item in metrics_list:
            config = metric_item.get("config", {})
            metric_type = config.get("type", "")

            if "ValueDrift" in metric_type:
                col_name = config.get("column")
                p_value = metric_item.get("value")
                threshold = config.get("threshold", 0.05)

                if col_name and p_value is not None:
                    # Drift detected if p_value < threshold
                    is_drifted = p_value < threshold
                    drifted_columns[col_name] = {
                        "drift_detected": bool(is_drifted),
                        "p_value": float(p_value),
                        "threshold": float(threshold),
                    }

        # Fallback if no ValueDrift metrics found (should not happen with DataDriftPreset)
        if not drifted_columns:
            print(f"[{model_type}] Warning: No ValueDrift metrics found.")

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
        active_models = ["xgboost", "rf", "elastic"]
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
