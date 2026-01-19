from mlflow.tracking import MlflowClient
from api_gateway.app.config import config
from api_gateway.app.constant.model_type import ModelType
import mlflow
import threading
from api_gateway.app.services.utils.thread_support import ThreadSupportMixin

LOADER_MAP = {
    ModelType.xgboost: mlflow.sklearn.load_model,
    # ModelType.catboost: mlflow.catboost.load_model,
    ModelType.rf: mlflow.sklearn.load_model,
    ModelType.elastic: mlflow.sklearn.load_model,
}


class MlflowModelManagementService(ThreadSupportMixin):
    def __init__(self):
        self.model_register_pattern = config.MLFLOW_MODEL_REGISTER_PATTERN
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        self.mlflow_client: MlflowClient = MlflowClient(
            config.MLFLOW_TRACKING_URI
        )
        self.refresh_interval = config.MLFLOW_MODEL_REFRESH_INTERVAL
        self._model_registry = {}
        self._lock = threading.Lock()
        super().__init__()

    def _switch_model(self, model_type, run_id):
        model_uri = f"models:/{self.model_register_pattern}_{model_type}@champion"

        loader = LOADER_MAP.get(model_type)

        if loader:
            model = loader(model_uri)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load Feature Engineering Pipeline
        pipeline = None
        try:
            local_path = self.mlflow_client.download_artifacts(
                run_id, "preprocessing/temp_engineer.pkl", dst_path="."
            )
            with open(local_path, "rb") as f:
                import pickle

                pipeline = pickle.load(f)
            # os.remove(local_path) # Optional: clean up
        except Exception as e:
            # Fallback or Log warning if no pipeline found (e.g. for old models)
            print(f"Warning: Could not load feature pipeline for {model_type}: {e}")

        return model, pipeline

    def _load_model(self):
        for model_type in ModelType:
            model_name = f"{self.model_register_pattern}_{model_type.value}"
            champion_version_details = self.mlflow_client.get_model_version_by_alias(
                model_name, "champion"
            )

            with self._lock:
                current = self._model_registry.get(model_type.value)
                if current and current["version"] == champion_version_details.version:
                    continue

            model, pipeline = self._switch_model(
                model_type=model_type.value, run_id=champion_version_details.run_id
            )

            print(
                f"Model {model_type.value} loaded version {champion_version_details.version}"
            )
            self._model_registry[model_type.value] = {
                "version": champion_version_details.version,
                "model": model,
                "pipeline": pipeline,
            }

    def thread_mixin_target_function(self):
        self._load_model()

    def get_model(self, model_type: ModelType):
        with self._lock:
            data = self._model_registry.get(model_type.value)
            if not data:
                # Attempt lazy load or raise error
                # For safety, let's trigger a single load attempt if missing
                self._load_model()
                data = self._model_registry.get(model_type.value)

            if not data:
                raise RuntimeError(
                    f"Model {model_type} is still loading or not found..."
                )
            return data

    def check_refresh(self):
        self._load_model()


mlflow_model_management_service = MlflowModelManagementService()
