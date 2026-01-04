import pandas as pd
import numpy as np
import json
from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseModelEvaluator(ABC):
    def __init__(self, thresholds: Dict[str, float] = None):
        """
        :param thresholds: Max/Min acceptable values for metrics.
                           Example: {'rmse': 500.0, 'r2': 0.7}
        """
        self.thresholds = thresholds or {}
        self.results_ = {}

    @abstractmethod
    def get_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Subclasses implement specific metrics (Regression vs Classification).
        """
        pass

    def validate_thresholds(self, metrics: Dict[str, float]) -> bool:
        """
        Checks if the calculated metrics meet the business requirements.
        """
        for metric, value in metrics.items():
            if metric in self.thresholds:
                # For error metrics (rmse, mae), lower is better
                if "error" in metric.lower() or "rmse" in metric.lower():
                    if value > self.thresholds[metric]:
                        return False
                # For score metrics (r2, accuracy), higher is better
                else:
                    if value < self.thresholds[metric]:
                        return False
        return True

    def run_evaluation(
        self, y_true: np.ndarray, y_pred: np.ndarray, metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        The main orchestration entry point for evaluation.
        """
        metrics = self.get_metrics(y_true, y_pred)
        passed = self.validate_thresholds(metrics)

        self.results_ = {
            "status": "PASSED" if passed else "FAILED",
            "metrics": metrics,
            "metadata": metadata or {},
            "timestamp": pd.Timestamp.now().isoformat(),
        }
        return self.results_

    def save_report(self, filepath: str):
        """Saves evaluation results as a JSON artifact."""
        with open(filepath, "w") as f:
            json.dump(self.results_, f, indent=4)
