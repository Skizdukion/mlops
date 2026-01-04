import mlflow
import mlflow.sklearn
import os
from typing import Dict, Any, Optional
from mlflow.tracking import MlflowClient


class MLflowModelRegistry:
    def __init__(
        self, experiment_name: str, tracking_uri: str = "http://localhost:5000"
    ):
        """
        :param experiment_name: Name of the project in MLflow (e.g., 'nyc-taxi-duration')
        """
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def register_model(
        self,
        model_name: str,
        model: Any,  # The actual model object (XGB, RF, etc.)
        eval_report: Dict[str, Any],
        feature_engineering: Any = None,  # The FeatureEngineer object
        params: Optional[Dict] = None,
    ):
        """
        Logs everything to MLflow in one atomic 'Run'.
        """
        with mlflow.start_run() as run:
            # 1. Log Hyperparameters
            if params:
                mlflow.log_params(params)

            # 2. Log Metrics
            mlflow.log_metrics(eval_report.get("metrics", {}))

            # 3. Log the Model (Native support for sklearn/xgboost)
            # This automatically handles serialization (no need for manual joblib)
            model_info = mlflow.sklearn.log_model(
                sk_model=model, artifact_path="model", registered_model_name=model_name
            )

            new_version = model_info.registered_model_version

            # 4. Log the Feature Engineer as a separate artifact
            if feature_engineering:
                # Save temporarily to log it
                feature_engineering.save_pipeline("temp_engineer.pkl")
                mlflow.log_artifact("temp_engineer.pkl", artifact_path="preprocessing")
                os.remove("temp_engineer.pkl")

            # 5. Tag the run status
            mlflow.set_tag("status", eval_report.get("status"))

            print(f"Model registered in MLflow! Run ID: {run.info.run_id}")
            return new_version

    def promote_to_production(
        self, model_name: str, candidate_version: str, metric_name: str = "rmse"
    ):
        """
        Compares the candidate model version against the current 'champion'.
        Promotes only if the candidate has a lower metric (e.g., RMSE).
        """
        client = MlflowClient()

        # 1. Get the candidate's metrics
        candidate_version_details = client.get_model_version(
            model_name, candidate_version
        )
        candidate_run = client.get_run(candidate_version_details.run_id)
        candidate_metric = candidate_run.data.metrics.get(metric_name)

        if candidate_metric is None:
            print(
                f"Aborting: Candidate version {candidate_version} has no metric '{metric_name}'."
            )
            return

        # 2. Try to get the current champion
        try:
            champion_version_details = client.get_model_version_by_alias(
                model_name, "champion"
            )
            champion_run = client.get_run(champion_version_details.run_id)
            champion_metric = champion_run.data.metrics.get(metric_name)

            print(f"Current Champion ({metric_name}): {champion_metric:.4f}")
            print(
                f"Candidate Version {candidate_version} ({metric_name}): {candidate_metric:.4f}"
            )

            # 3. Comparison Logic (Lower is better for RMSE)
            if candidate_metric < champion_metric:
                client.set_registered_model_alias(
                    model_name, "champion", candidate_version
                )
                print(
                    f"✅ Success: Candidate is better. Promoted version {candidate_version} to 'champion'."
                )
            else:
                print(
                    f"❌ Rejected: Candidate {candidate_metric:.4f} is not better than Champion {champion_metric:.4f}."
                )

        except mlflow.exceptions.RestException:
            # This triggers if no 'champion' alias exists yet
            print(
                "No existing 'champion' found. Promoting candidate as the first champion."
            )
            client.set_registered_model_alias(model_name, "champion", candidate_version)
