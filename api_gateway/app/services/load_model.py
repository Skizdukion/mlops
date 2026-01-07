from mlflow.tracking import MlflowClient
from api_gateway.app.config import config
from api_gateway.app.constant.model_type import ModelType
import mlflow
import threading

LOADER_MAP = {
    ModelType.xgboost: mlflow.xgboost.load_model,
    ModelType.catboost: mlflow.catboost.load_model,
    ModelType.rf: mlflow.sklearn.load_model,
    ModelType.elastic: mlflow.sklearn.load_model,
}


class MlflowModelManagementService:
    def __init__(self):
        self.experiment_name = config.MLFLOW_EXPERIMENT_NAME
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
        self.mlflow_client: MlflowClient = MlflowClient(config.MLFLOW_TRACKING_URI)
        self.refresh_interval = config.MLFLOW_MODEL_REFRESH_INTERVAL
        self._model_registry = {}
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def _switch_model(self, model_type):
        model_uri = f"models:/{self.experiment_name}_{model_type}@champion"

        loader = LOADER_MAP.get(model_type)

        if loader:
            model = loader(model_uri)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return model

    def _load_model(self):
        for model_type in ModelType:
            champion_version_details = self.mlflow_client.get_model_version_by_alias(
                model_type.value, "champion"
            )

            with self._lock:
                current = self._model_registry.get(model_type.value)
                if current and current["version"] == champion_version_details.version:
                    continue

            model = self._switch_model(model_type=model_type.value)

            print(
                f"Model {model_type.value} loaded version {champion_version_details.version}"
            )
            self._model_registry[model.value] = {
                "version": champion_version_details.version,
                "model": model,
            }

    def get_model(self, model_type: ModelType):
        with self._lock:
            data = self._model_registry.get(model_type.value)
            if not data:
                raise RuntimeError(f"Model {model_type} is still loading...")
            return data["model"]

    def check_refresh(self):
        self._load_model()

    def refresh_loop(self):
        while not self._stop_event.is_set():
            try:
                self._load_model()
            except Exception:
                print("Model refresh failed")

            self._stop_event.wait(self.refresh_interval)

    def start(self):
        self._thread = threading.Thread(target=self.refresh_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)


mlflow_model_management_service = MlflowModelManagementService()
