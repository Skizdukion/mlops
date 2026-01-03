import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class NumericalMedianFiller(FeatureTransformer):
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.medians_ = {}

    def fit(self, df: pd.DataFrame):
        """Learn the median for each column from training data."""
        for col in self.columns:
            if col in df.columns:
                self.medians_[col] = df[col].median()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs using the medians learned during fit."""
        for col, median_val in self.medians_.items():
            if col in df.columns:
                df[col] = df[col].fillna(median_val)
        return df
