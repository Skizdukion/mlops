from abc import ABC, abstractmethod
import pandas as pd


class FeatureTransformer(ABC):
    """
    Abstract Base Class for all Feature Engineering steps.
    Ensures that every step has a fit and transform logic.
    """

    @abstractmethod
    def fit(self, df: pd.DataFrame):
        """
        Learn logic from the training data (e.g., calculate mean,
        identify categories).
        """
        pass

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the learned logic to a DataFrame.
        """
        pass

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standard helper to do both in one go."""
        self.fit(df)
        return self.transform(df)
