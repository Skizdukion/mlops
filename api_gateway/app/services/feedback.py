from api_gateway.app.storage.repository import repo
from api_gateway.app.models.domain import Feedback
from api_gateway.app.routes.dto.feedback import NycDurationFeedbackRequest


class NycDurationFeedbackService:
    def __init__(self):
        pass

    def feedback(self, request: NycDurationFeedbackRequest):
        # store feedback
        feedback_doc = Feedback(**request.dict())
        repo.save_feedback(feedback_doc)

        return


feedback_service = NycDurationFeedbackService()
