import sys
import time
import pandas as pd
import whylogs as why
from whylogs.core import DatasetProfileView
from pathlib import Path
from datetime import datetime
from sqlmodel import Session, create_engine, text

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

# Global Cache: {model_type: {'profile': DatasetProfileView, 'checksum': (count, max_ts)}}
REFERENCE_CACHE = {}


def get_reference_profile_checksum(model_type: str):
    """
    Returns a tuple (count, max_created_at) for the current reference data.
    Used to determine if we need to reload the cache.
    """
    query = text(
        f"""
        SELECT COUNT(*), MAX(f.created_at)
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = :model_type
          AND f.is_current_train = True
    """
    )

    with Session(engine) as session:
        result = session.execute(query, {"model_type": model_type}).fetchone()
        if result:
            return (
                result[0],
                str(result[1]),
            )  # Convert TS to string for loose comparison
    return (0, None)


def get_reference_profile(model_type: str) -> DatasetProfileView:
    """
    Returns cached profile if valid, otherwise loads from DB, profiles it, and caches.
    """
    current_checksum = get_reference_profile_checksum(model_type)

    # Check cache
    cached = REFERENCE_CACHE.get(model_type)
    if cached:
        # Check if checksum matches and we actually have data (count > 0)
        # Handle case where cached checksum might have None if empty previously
        if cached["checksum"] == current_checksum and current_checksum[0] > 0:
            return cached["profile"]
        elif cached["checksum"][0] == 0 and current_checksum[0] == 0:
            return None

    # Cache Miss or Stale -> Reload
    print(
        f"[{model_type}] Reference cache invalid/missing {current_checksum}. Loading from DB..."
    )
    df = load_reference_data(model_type)

    if df.empty:
        return None

    print(f"[{model_type}] Profiling {len(df)} reference rows...")
    profile_view = why.log(df).profile().view()

    REFERENCE_CACHE[model_type] = {
        "profile": profile_view,
        "checksum": current_checksum,
    }

    return profile_view


def load_reference_data(model_type: str) -> pd.DataFrame:
    """Load training data marked as current reference."""
    # Note: this loads full data. Helper for get_reference_profile.
    query = f"""
        SELECT 
            p.tpep_pickup_datetime, p.predicted_duration as prediction, f.duration as target
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = '{model_type}'
          AND f.is_current_train = True
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


def calculate_drift(ref_profile_view, current_df):
    if ref_profile_view is None or current_df.empty:
        return None

    # Profile current data
    cur_profile_view = why.log(current_df).profile().view()

    # Calculate Drift Scores
    # Note: requiring whylogs[viz] or manual calculation
    try:
        from whylogs.viz.drift.column_drift_algorithms import calculate_drift_scores

        scores = calculate_drift_scores(
            target_view=cur_profile_view,
            reference_view=ref_profile_view,
            with_thresholds=True,
        )
        return scores
    except ImportError:
        print("whylogs.viz not available. Cannot calculate drift scores.")
        return {}


def save_drift_metrics(drift_scores, model_type):
    if not drift_scores:
        return

    try:
        drifted_cols_count = 0
        total_drift_score = 0.0
        drifted_details = {}

        for col, score_info in drift_scores.items():
            is_drifted = score_info.get("drift_category") == "DRIFT"
            p_val = score_info.get("p_value", 1.0)

            if is_drifted:
                drifted_cols_count += 1
                drifted_details[col] = p_val

            total_drift_score += 1 - p_val

        avg_drift_score = total_drift_score / len(drift_scores) if drift_scores else 0.0
        drift_share = drifted_cols_count / len(drift_scores) if drift_scores else 0.0

        metric_record = DatadriftsMetrics(
            timestamp=datetime.utcnow(),
            model_type=model_type,
            data_drift=drift_share,
            drifted_columns=drifted_details,
        )

        with Session(engine) as session:
            session.add(metric_record)
            session.commit()
            print(
                f"[{model_type}] Drift saved: Share={drift_share:.4f}, DriftedCols={drifted_cols_count}"
            )

    except Exception as e:
        print(f"Error saving drift for {model_type}: {e}")


def main():
    print("Data Drift Worker started (Whylogs with Caching)...")

    while True:
        # active_models = ["catboost", "xgboost"]
        active_models = ["xgboost"]

        for model in active_models:
            print(f"Checking drift for {model}...")

            # 1. Get Reference Profile (Cached)
            ref_profile = get_reference_profile(model)
            if ref_profile is None:
                print(f"[{model}] No reference data found. Skipping.")
                continue

            # 2. Load Current Data
            curr_df = load_current_data(model, CURRENT_DATA_LIMIT)

            # 3. Calculate & Save
            if not curr_df.empty:
                try:
                    drift_result = calculate_drift(ref_profile, curr_df)
                    if drift_result:
                        save_drift_metrics(drift_result, model)
                except Exception as e:
                    print(f"Drift calc error: {e}")
            else:
                print(f"[{model}] No current data. Skipping.")

        time.sleep(LOOP_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
