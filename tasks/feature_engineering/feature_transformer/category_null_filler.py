import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class CategoricalNullFiller(FeatureTransformer):
    def __init__(self, columns: list[str], fill_value: str = "Unknown"):
        self.columns = columns
        self.fill_value = fill_value

    def fit(self, df: pd.DataFrame):
        # Stateless: Nothing to learn
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in self.columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(self.fill_value)
        return df_copy
