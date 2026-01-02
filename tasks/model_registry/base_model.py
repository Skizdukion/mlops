import mlflow
import mlflow.sklearn
import os
import json
from typing import Dict, Any, Optional


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
        engineer: Any = None,  # The FeatureEngineer object
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
            mlflow.sklearn.log_model(
                sk_model=model, artifact_path="model", registered_model_name=model_name
            )

            # 4. Log the Feature Engineer as a separate artifact
            if engineer:
                # Save temporarily to log it
                engineer.save_pipeline("temp_engineer.pkl")
                mlflow.log_artifact("temp_engineer.pkl", artifact_path="preprocessing")
                os.remove("temp_engineer.pkl")

            # 5. Tag the run status
            mlflow.set_tag("status", eval_report.get("status"))

            print(f"Model registered in MLflow! Run ID: {run.info.run_id}")
            return run.info.run_id

    def promote_to_production(self, model_name: str, version: str):
        """
        Uses MLflow Aliases (Modern way) to mark a model as Production.
        """
        client = mlflow.tracking.MlflowClient()
        client.set_registered_model_alias(model_name, "champion", version)
        print(f"Model {model_name} version {version} is now Aliased as 'champion'")
