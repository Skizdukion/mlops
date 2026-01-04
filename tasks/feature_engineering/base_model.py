from abc import ABC
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
        IMPORTANT: This method modifies the input DataFrame in-place.
        If you need to preserve the original, copy it before calling fit.
        """
        for transformer in self.transformers:
            transformer.fit(df)
            # We must transform so the NEXT transformer
            # sees the data as it will appear in the real pipeline.
            df = transformer.transform(df)

        self._is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies all learned transformations to the data.
        IMPORTANT: This method modifies the input DataFrame in-place.
        If you need to preserve the original, copy it before calling transform.
        """
        if not self._is_fitted:
            raise RuntimeError("Engineer must be fitted before calling transform.")

        for transformer in self.transformers:
            df = transformer.transform(df)

        return df

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
