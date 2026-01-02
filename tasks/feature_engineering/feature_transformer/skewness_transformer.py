import numpy as np
import pandas as pd

from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class SkewnessTransformer(FeatureTransformer):
    def __init__(self, columns: list[str], method: str = "log"):
        """
        :param method: 'log' (right skew), 'sqrt' (right skew), or 'exp' (left skew)
        """
        self.columns = columns
        self.method = method

    def fit(self, df: pd.DataFrame):
        return self  # Stateless, though Box-Cox would require a fit

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col in self.columns:
            if col in df_copy.columns:
                if self.method == "log":
                    # np.log1p handles log(0) by doing log(1+x)
                    df_copy[col] = np.log1p(df_copy[col])
                elif self.method == "sqrt":
                    df_copy[col] = np.sqrt(df_copy[col])
                elif self.method == "exp":
                    df_copy[col] = np.exp(df_copy[col])
        return df_copy
