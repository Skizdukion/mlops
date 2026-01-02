import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class OutlierClipper(FeatureTransformer):
    def __init__(
        self,
        columns: list[str],
        lower_percentile: float = 0.01,
        upper_percentile: float = 0.99,
    ):
        self.columns = columns
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.limits_ = {}  # To store learned bounds

    def fit(self, df: pd.DataFrame):
        """Calculate the bounds for each column."""
        for col in self.columns:
            lower = df[col].quantile(self.lower_percentile)
            upper = df[col].quantile(self.upper_percentile)
            self.limits_[col] = (lower, upper)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the clipping using stored bounds."""
        df_copy = df.copy()
        for col, (lower, upper) in self.limits_.items():
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].clip(lower=lower, upper=upper)
        return df_copy
