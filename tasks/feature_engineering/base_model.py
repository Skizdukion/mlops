from abc import ABC, abstractmethod
import pandas as pd
import pickle
from typing import List
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class FeatureEngineer(ABC):
    def __init__(self, transformers: List[FeatureTransformer] = None):
        """
        :param transformers: A list of FeatureTransformer instances.
        """
        self.transformers = transformers or []
        self._is_fitted = False

    def add_transformer(self, transformer: FeatureTransformer):
        """Allows adding transformers dynamically before fitting."""
        self.transformers.append(transformer)

    def fit(self, df: pd.DataFrame):
        """
        Sequentially fits all transformers on the training data.
        Note: We use a copy to avoid side effects during the fit phase.
        """
        temp_df = df.copy()
        for transformer in self.transformers:
            transformer.fit(temp_df)
            # We must transform the temp_df so the NEXT transformer
            # sees the data as it will appear in the real pipeline.
            temp_df = transformer.transform(temp_df)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies all learned transformations to the data."""
        if not self._is_fitted:
            raise RuntimeError("Engineer must be fitted before calling transform.")

        df_transformed = df.copy()
        for transformer in self.transformers:
            df_transformed = transformer.transform(df_transformed)

        return df_transformed

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Helper for training phase."""
        return self.fit(df).transform(df)

    # --- Orchestration Utilities ---

    def save_pipeline(self, filepath: str):
        """Saves the fitted state to disk for production use."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)
        print(f"Pipeline saved to {filepath}")

    @classmethod
    def load_pipeline(cls, filepath: str):
        """Loads a fitted pipeline for inference."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
