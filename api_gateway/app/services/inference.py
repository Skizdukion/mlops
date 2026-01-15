from api_gateway.app.services.load_model import mlflow_model_management_service
from api_gateway.app.storage.repository import repo
from alembic_model.models.domain import Prediction
import pandas as pd
from api_gateway.app.constant.model_type import ModelType


class NycDurationInferenceService:
    def __init__(self):
        pass

    def inference(
        self,
        tpep_pickup_datetime,
        passenger_count,
        pulocationid,
        dolocationid,
        model_type: ModelType,
    ):
        # 1. Prepare Input Data as DataFrame
        input_data = {
            "tpep_pickup_datetime": [tpep_pickup_datetime],
            "passenger_count": [passenger_count],
            "pulocationid": [pulocationid],
            "dolocationid": [dolocationid],
        }
        df = pd.DataFrame(input_data)

        # 2. Get Model and Pipeline
        model_data = mlflow_model_management_service.get_model(model_type=model_type)
        model = model_data["model"]
        pipeline = model_data.get("pipeline")

        # 3. Apply Feature Engineering
        df = pipeline.transform(df)

        df = df.drop(columns=["tpep_pickup_datetime"])
        df = df.drop(columns=["pulocationid"])
        df = df.drop(columns=["dolocationid"])

        # 4. Inference
        # model.predict usually returns a numpy array or list
        predictions = model.predict(df)
        predicted_duration = float(predictions[0])

        # Store inference to db
        prediction_doc = Prediction(
            tpep_pickup_datetime=tpep_pickup_datetime,
            passenger_count=passenger_count,
            pulocationid=pulocationid,
            dolocationid=dolocationid,
            model_type=model_type.value,
            predicted_duration=predicted_duration,
        )

        prediction_id = repo.save_inference(prediction_doc)

        # Return inference
        return {"prediction_id": prediction_id, "est_duration": predicted_duration}


inference_service = NycDurationInferenceService()
