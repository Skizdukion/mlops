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
        for col in self.columns:
            if col in df.columns:
                if self.method == "log":
                    # np.log1p handles log(0) by doing log(1+x)
                    # However, it cannot handle negative values, so clip to 0
                    df[col] = np.log1p(np.maximum(df[col], 0))
                elif self.method == "sqrt":
                    # sqrt cannot handle negative values, so clip to 0
                    df[col] = np.sqrt(np.maximum(df[col], 0))
                elif self.method == "exp":
                    df[col] = np.exp(df[col])
        return df
