from api_gateway.app.storage.repository import repo
from api_gateway.app.models.domain import Feedback


class NycDurationInferenceService:
    def __init__(self):
        pass

    def feedback(self, prediction_id, dolocationid, duration, model_type):
        # store feedback
        feedback_doc = Feedback(
            prediction_id=prediction_id,
            dolocationid=dolocationid,
            duration=duration,
        )
        repo.save_feedback(feedback_doc)
        return
