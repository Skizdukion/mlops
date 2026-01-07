
from api_gateway.app.services.load_model import mlflow_model_management_service

class NycDurationInferenceService:
    def __init__(self):
        pass

    def inference(self, tpep_pickup_datetime, passenger_count, pulocationid, dolocationid, model_type):
        # Generate id
        # Get feature engineering base on model type -> proceed it
        # Get model from mlflow_model_management_service
        # Inference with model
        # Store inference to db
        # Return inference
        pass

