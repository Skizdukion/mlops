import threading
from api_gateway.app.services.utils.thread_support import ThreadSupportMixin
from api_gateway.app.config import config
from api_gateway.app.storage.repository import repo
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Fetch last_row_count from db with created_at desc
# Compute on y_pred and y_groundtruh with metrics
# If below threshold, send notification and possibly trigger new training pipeline
class NycDurationMonitoringService(ThreadSupportMixin):
    def __init__(self):
        self.refresh_interval = config.MONITORING_METRICS_INTERVAL
        self.last_row_count = config.METRICS_LAST_ROW_FETCH
        self._lock = threading.Lock()
        self.rmse_thresh_hold = 8.0
        super().__init__()

    def get_recent_metrics(self):
        rows = repo.get_last_preds_with_feedback(self.last_row_count)

        if len(rows) == 0:
            return (0, 0, 0)

        y = [row.feedback.duration for row in rows]

        # Extract y_pred (predicted values from Prediction)
        y_pred = [row.predicted_duration for row in rows]

        # Calculate MSE
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        return (mse, mae, r2)

    def thread_mixin_target_function(self):
        (mse, mae, r2) = self.get_recent_metrics()

        if mse**0.5 > self.rmse_thresh_hold:
            print("WARNING: MSE below threshold, send signal to telegram/slack")

        return


monitoring_service = NycDurationMonitoringService()
