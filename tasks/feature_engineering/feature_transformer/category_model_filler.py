import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class CategoricalModeFiller(FeatureTransformer):
    def __init__(self, columns: list[str]):
        self.columns = columns
        self.modes_ = {}  # To store learned modes

    def fit(self, df: pd.DataFrame):
        """Learn the mode for each column from training data."""
        for col in self.columns:
            # value_counts() sorts by frequency descending; index[0] is the mode
            mode_val = df[col].mode()
            if not mode_val.empty:
                self.modes_[col] = mode_val[0]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaNs using the modes learned during fit."""
        for col, mode_value in self.modes_.items():
            if col in df.columns:
                df[col].fillna(mode_value, inplace=True)
        return df
