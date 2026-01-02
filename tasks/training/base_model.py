import pandas as pd
import numpy as np
import joblib
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

class BaseModelTrainer(ABC):
    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        :param model_params: Dictionary of hyperparameters for the specific model.
        """
        self.model_params = model_params or {}
        self.model = self.init_model()
        self.feature_importance_ = None

        # Configure logging for the trainer
        self.logger = logging.getLogger(self.__class__.__name__)
        logging.basicConfig(level=logging.INFO)

    @abstractmethod
    def init_model(self) -> Any:
        """
        Subclasses must implement this to return the specific
        algorithm instance (e.g., return XGBRegressor()).
        """
        pass

    @abstractmethod
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        """
        Subclasses must implement the fit logic.
        """
        pass

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Standardized prediction method.
        """
        if self.model is None:
            raise RuntimeError("Model has not been initialized.")
        return self.model.predict(X)

    def save_model(self, filepath: str):
        """
        Saves the model artifact to disk.
        """
        try:
            joblib.dump(self.model, filepath)
            self.logger.info(f"Model artifact successfully saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            raise

    def load_model(self, filepath: str):
        """
        Loads a model artifact from disk.
        """
        try:
            self.model = joblib.load(filepath)
            self.logger.info(f"Model artifact loaded from {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def get_feature_importance(self) -> pd.Series:
        """
        Generic helper to extract feature importance if the model supports it.
        """
        if hasattr(self.model, "feature_importances_"):
            return pd.Series(self.model.feature_importances_)
        elif hasattr(self.model, "coef_"):
            return pd.Series(self.model.coef_)
        else:
            self.logger.warning("Model does not support standard feature importance.")
            return pd.Series()
