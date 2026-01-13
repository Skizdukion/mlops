import sys
import os
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlmodel import Session, create_engine
from evidently import Report, DataDefinition, Regression, Dataset
from evidently.presets import RegressionPreset


# Add project root to sys.path to allow imports from api_gateway and monitoring
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from monitoring.models import MonitoringMetrics
from api_gateway.app.config import config


# Setup Database Engine
# We can use the same DATABASE_URL from api_gateway config
engine = create_engine(config.DATABASE_URL)

LOOP_INTERVAL_SECONDS = 60
ROW_LIMIT = 500


def load_data(model_type: str, limit=1000) -> pd.DataFrame:
    """
    Fetch recent predictions and feedback from the database for a specific model.
    """
    query = f"""
        SELECT 
            p.tpep_pickup_datetime,
            p.passenger_count,
            p.pulocationid,
            p.dolocationid,
            p.predicted_duration as prediction,
            f.duration as target
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = '{model_type}'
        ORDER BY p.tpep_pickup_datetime DESC
        LIMIT {limit}
    """

    try:
        df = pd.read_sql(query, engine)
        return df
    except Exception as e:
        print(f"Error loading data for {model_type}: {e}")
        return pd.DataFrame()


def get_active_models():
    """Fetch distinct model types from predictions."""
    query = "SELECT DISTINCT model_type FROM nyc_duration_inferences"
    try:
        df = pd.read_sql(query, engine)
        if not df.empty:
            return df["model_type"].tolist()
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def calculate_metrics(df):
    if df.empty:
        return None

    # User requested to focus on RMSE, MAE, R2 only. No need for split or reference/current comparison for drift.
    # We will compute regression quality on the fetched batch.

    definition = DataDefinition(
        regression=[Regression(target="target", prediction="prediction")]
    )

    report = Report(
        metrics=[
            # DataDriftPreset(), # Commented out as per user request
            RegressionPreset()
        ]
    )

    current_ds = Dataset.from_pandas(df, data_definition=definition)

    snapshot = report.run(reference_data=None, current_data=current_ds)

    results = snapshot.dict()

    metrics_results = {m["metric_name"]: m["value"] for m in results["metrics"]}
    return metrics_results


def save_metrics(metrics_data, model_type):
    if not metrics_data:
        return

    data_drift = None
    rmse = 0.0
    r2 = 0.0
    mae = 0.0
    drifted_cols = {}

    try:
        # Provide default access since Structure can vary
        # RegresionPreset > RegressionQualityMetric

        # We can iterate or try to direct access if we know structure.
        # But iterating is safer given presets might nest things differently or flatten them.

        for metric_name, value in metrics_data.items():
            if "RMSE" in metric_name:
                rmse = value
            if "R2Score" in metric_name:
                r2 = value
            if "MAE" in metric_name:
                mae = value["mean"]

        # Create DB object
        metric_record = MonitoringMetrics(
            timestamp=datetime.utcnow(),
            model_type=model_type,
            data_drift=data_drift,
            rmse=rmse,
            r2=r2,
            mae=mae,
            drifted_columns=drifted_cols,
        )

        with Session(engine) as session:
            session.add(metric_record)
            session.commit()
            print(
                f"[{model_type}] Metrics saved: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
            )

    except Exception as e:
        print(f"Error saving metrics for {model_type}: {e}")


def main():
    print("Worker started...")

    # Optional: Run migrations here or assume they are run externally
    # For now, we assume migrations are run via 'alembic upgrade head' in entrypoint or manually.

    while True:
        print("Starting monitoring cycle...")

        models = get_active_models()
        print(f"Active models found: {models}")

        for model in models:
            print(f"Processing model: {model}")
            df = load_data(model, ROW_LIMIT)

            if not df.empty:
                print(f"Data loaded for {model}: {len(df)} rows")

                result = calculate_metrics(df)
                if result:
                    save_metrics(result, model)
                else:
                    print(f"Could not calculate metrics for {model}.")

            else:
                print(f"No data found for {model}.")

        print(f"Sleeping for {LOOP_INTERVAL_SECONDS} seconds...")
        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
