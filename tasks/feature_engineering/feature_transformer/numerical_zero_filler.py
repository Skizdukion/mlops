import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class NumericalZeroFiller(FeatureTransformer):
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, df: pd.DataFrame):
        # Stateless: No calculation needed from training data
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in self.columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].fillna(0)
        return df_copy
