import sys
import os
from pathlib import Path
import pandas as pd
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.presets import DataDriftPreset, RegressionPreset
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to sys.path to allow imports from api_gateway
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from api_gateway.app.config import config  # noqa: E402
from api_gateway.app.storage.repository import repo  # noqa: E402
from api_gateway.app.models.domain import Prediction, Feedback  # noqa: E402


def get_data(limit: int = 1000) -> pd.DataFrame:
    """Fetch recent predictions with feedback from DB."""
    # Using the repository method (we might need to ensure it returns what we want or query directly)
    # repo.get_last_preds_with_feedback returns list[Prediction] with .feedback loaded.

    # However, to use Evidently, we need a flat DataFrame.
    # Let's query and flatten.

    preds = repo.get_last_preds_with_feedback(limit)

    data = []
    for p in preds:
        if p.feedback:
            # We have both prediction and ground truth
            row = {
                "tpep_pickup_datetime": p.tpep_pickup_datetime,
                "passenger_count": p.passenger_count,
                "pulocationid": p.pulocationid,
                "dolocationid": p.dolocationid,
                "model_type": p.model_type,
                "prediction": p.predicted_duration,
                "target": p.feedback.duration,
            }
            data.append(row)

    df = pd.DataFrame(data)
    return df


def run_scoring(row_limit: int = 1000):
    print(f"Fetching last {row_limit} rows with feedback...")
    df = get_data(row_limit)

    if df.empty:
        print("No paired data found for scoring.")
        return

    print(f"Data shape: {df.shape}")

    # Evidently Report
    report = Report(
        metrics=[
            DataDriftPreset(),
            RegressionPreset(),
        ]
    )

    # We need a reference dataset for drift calculations to be meaningful.
    # Typically we compare 'current' batch vs 'reference' batch.
    # For this script, maybe we can split the fetched data? Or just calculate quality if no reference?
    # Drift requires reference.
    # If we only have "recent n", maybe we split it: first half reference, second half current?
    # Or typically reference comes from training data.
    # For now, to demonstrate functionality, I will split the data 50/50 if enough rows,
    # or just run Quality metrics if not enough for split.
    # User asked "check recent last n", usually implies current batch.
    # Without reference, drift is hard.
    # I'll create a dummy reference from the first half of data for demonstration purposes,
    # or if the user provided a training set reference file, I'd use that.
    # Let's split it.

    cutoff = len(df) // 2
    reference = df.iloc[cutoff:]
    current = df.iloc[:cutoff]  # Last N are at the top usually if sorted desc?
    # Repo sorts desc. So index 0 is newest.
    # So 'current' (newest) is top. 'reference' (older) is bottom.

    # Evidently expects: reference (old), current (new)
    # If df is sorted DESC by time (newest first):
    # current = df.iloc[:cutoff]  (Newer)
    # reference = df.iloc[cutoff:] (Older)

    print("\nCalculated Metrics (Reference: older half, Current: newer half):")

    definition = DataDefinition(
        regression=[Regression(target="target", prediction="prediction")]
    )

    reference_ds = Dataset.from_pandas(reference, data_definition=definition)
    current_ds = Dataset.from_pandas(current, data_definition=definition)

    snapshot = report.run(reference_data=reference_ds, current_data=current_ds)

    # Print simplified results
    results = snapshot.dict()

    # print(results)
    # Parse parameters from metrics
    metrics_results = {m["metric_name"]: m["value"] for m in results["metrics"]}

    # Find metrics by name (sometimes they have parameters in name, so we search)
    drift_share = 0.0
    rmse = 0.0
    r2 = 0.0
    mae = 0.0

    for metric_name, value in metrics_results.items():
        if "DriftedColumnsCount" in metric_name:
            drift_share = value["share"]
        if "RMSE" in metric_name:
            rmse = value
        if "R2Score" in metric_name:
            r2 = value
        if "MAE" in metric_name:
            mae = value["mean"]

    print(f"Share of Drifted Columns: {drift_share}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    # Save report
    output_path = "monitoring/report.html"
    snapshot.save_html(output_path)
    print(f"\nDetailed HTML report saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit", type=int, default=1000, help="Number of rows to check"
    )
    args = parser.parse_args()

    run_scoring(args.limit)
