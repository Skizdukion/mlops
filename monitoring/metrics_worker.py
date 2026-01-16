import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from sqlmodel import Session, create_engine, select
from evidently import Report, DataDefinition, Regression, Dataset
from evidently.presets import RegressionPreset

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
from monitoring.models import MonitoringMetrics
from alembic_model.config import config

# Setup Database Engine
engine = create_engine(config.DATABASE_URL)

RMSE_THRESHOLD = 8.0  # Example threshold for retraining
RETRAINING_DEPLOYMENT_NAME = "NYC Taxi Duration Training Pipeline/regular-training"  # Update with actual deployment name


def trigger_retraining_flow(model_type: str, reason: str):
    """Triggers the Prefect flow for retraining."""
    print(f"[TRIGGER] Triggering retraining for {model_type}. Reason: {reason}")
    try:
        # In a real scenario, correct deployment name is required.
        # Assuming parameters allow 'from_db' override.
        # run_deployment(name=RETRAINING_DEPLOYMENT_NAME, parameters={"model_type": model_type, "from_db": True})
        pass
    except Exception as e:
        print(f"Error triggering retraining: {e}")


def load_data(model_type: str, window_size: str) -> pd.DataFrame:
    """
    Fetch data based on window size.
    window_size can be: '10k', '1d', '7d'
    """
    query = ""
    if window_size == "10k":
        # Get last 10000 rows
        limit = 10000
        query = f"""
            SELECT 
                p.tpep_pickup_datetime, p.predicted_duration as prediction, f.duration as target, f.created_at
            FROM nyc_duration_inferences p
            JOIN nyc_duration_feedback f ON p.id = f.prediction_id
            WHERE p.model_type = '{model_type}'
            ORDER BY p.tpep_pickup_datetime DESC
            LIMIT {limit}
        """
    elif window_size == "1d":
        # Get rows from last 24 hours
        query = f"""
            SELECT 
                p.tpep_pickup_datetime, p.predicted_duration as prediction, f.duration as target
            FROM nyc_duration_inferences p
            JOIN nyc_duration_feedback f ON p.id = f.prediction_id
            WHERE p.model_type = '{model_type}'
              AND f.created_at >= NOW() - INTERVAL '1 DAY'
            ORDER BY p.tpep_pickup_datetime DESC
        """
    elif window_size == "7d":
        # Get rows from last 7 days
        query = f"""
            SELECT 
                p.tpep_pickup_datetime, p.predicted_duration as prediction, f.duration as target
            FROM nyc_duration_inferences p
            JOIN nyc_duration_feedback f ON p.id = f.prediction_id
            WHERE p.model_type = '{model_type}'
              AND f.created_at >= NOW() - INTERVAL '7 DAYS'
            ORDER BY p.tpep_pickup_datetime DESC
        """

    try:
        df = pd.read_sql(query, engine)

        # specific check for 10k window to ensure we have enough data
        if window_size == "10k" and len(df) < 10000:
            print(f"[{model_type} - {window_size}] Not enough data: {len(df)}/10000")
            return pd.DataFrame()  # Return empty if not enough

        return df
    except Exception as e:
        print(f"Error loading data for {model_type} ({window_size}): {e}")
        return pd.DataFrame()


def get_last_processed_info(model_type: str, window_size: str):
    """Check when this window was last successfully processed"""
    try:
        with Session(engine) as session:
            statement = (
                select(MonitoringMetrics)
                .where(
                    MonitoringMetrics.model_type == model_type,
                    MonitoringMetrics.report_name == window_size,
                )
                .order_by(MonitoringMetrics.timestamp.desc())
                .limit(1)
            )
            result = session.exec(statement).first()
            if result:
                return result.timestamp
    except Exception as e:
        print(f"Error checking last info: {e}")
    return None


def calculate_metrics(df):
    if df.empty:
        return None

    definition = DataDefinition(
        regression=[Regression(target="target", prediction="prediction")]
    )
    report = Report(metrics=[RegressionPreset()])
    current_ds = Dataset.from_pandas(df, data_definition=definition)
    snapshot = report.run(reference_data=None, current_data=current_ds)
    results = snapshot.dict()
    metrics_results = {m["metric_name"]: m["value"] for m in results["metrics"]}
    return metrics_results


def save_metrics(metrics_data, model_type, window_size):
    if not metrics_data:
        return

    rmse = 0.0
    r2 = 0.0
    mae = 0.0

    try:
        for metric_name, value in metrics_data.items():
            if "RMSE" in metric_name:
                rmse = value
            if "R2Score" in metric_name:
                r2 = value
            if "MAE" in metric_name:
                mae = value["mean"]

        metric_record = MonitoringMetrics(
            timestamp=datetime.utcnow(),
            model_type=model_type,
            rmse=rmse,
            r2=r2,
            mae=mae,
            report_name=window_size,  # Storing window type here
        )

        with Session(engine) as session:
            session.add(metric_record)
            session.commit()
            print(
                f"[{model_type} - {window_size}] Metrics saved: RMSE={rmse:.4f}, MAE={mae:.4f}"
            )

        # Check for Retraining Trigger (Only on 1d window)
        if window_size == "1d" and rmse > RMSE_THRESHOLD:
            trigger_retraining_flow(model_type, f"RMSE {rmse:.2f} > {RMSE_THRESHOLD}")

    except Exception as e:
        print(f"Error saving metrics for {model_type}: {e}")


def process_window(model_type: str, window_size: str):
    print(f"Processing {model_type} window: {window_size}")

    # 1. Load Data
    df = load_data(model_type, window_size)
    if df.empty:
        return

    metrics = calculate_metrics(df)
    if metrics:
        save_metrics(metrics, model_type, window_size)


def main():
    print("Metrics Worker started...")

    # Scheduling config
    # 10k window -> Run every 10 mins
    # 1d window -> Run every 15 mins (test mode) / 1 day (prod)
    # 7d window -> Run every 15 mins (test mode) / 7 days (prod)

    # We will use a simple counter or timestamp check in the loop
    last_run = {"10k": datetime.min, "1d": datetime.min, "7d": datetime.min}

    intervals = {
        "10k": 10 * 60,  # 10 minutes
        "1d": 15 * 60,  # 15 minutes (Test mode) - normally 24h
        "7d": 15 * 60,  # 15 minutes (Test mode) - normally 7d
    }

    while True:
        now = datetime.utcnow()
        active_models = ["xgboost", "rf", "elastic"]  # Ideally fetch from DB

        for window in ["10k", "1d", "7d"]:
            if (now - last_run[window]).total_seconds() >= intervals[window]:
                print(f"--- Running {window} check ---")
                for model in active_models:
                    process_window(model, window)
                last_run[window] = now

        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    main()
