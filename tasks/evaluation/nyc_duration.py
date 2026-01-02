import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tasks.evaluation.base_model import BaseModelEvaluator


class NYCDurationEvaluator(BaseModelEvaluator):
    """
    Evaluator specifically for NYC Trip Duration.
    Expects target and predictions in minutes (or seconds).
    """

    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """
        Calculates the core regression suite for NYC trip duration.
        """
        # RMSE calculation
        mse = mean_squared_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))

        # Logging metrics
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        return {
            "rmse": rmse,  # Main Primary Metric
            "mae": mae,  # Average physical error
            "r2": r2,  # Variance explained (0.0 to 1.0)
            "mse": float(mse),
        }

    def get_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Extra logic: NYC specific. Check how many predictions were
        off by more than 10 minutes (significant for riders).
        """
        errors = np.abs(y_true - y_pred)
        over_10_min = np.mean(errors > 10) * 100

        return {"pct_errors_over_10min": float(over_10_min)}
