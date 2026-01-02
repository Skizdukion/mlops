import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer

class ValueClipper(FeatureTransformer):
    def __init__(self, limits: dict[str, tuple[float | None, float | None]]):
        """
        :param limits: dict like {'column_name': (min_val, max_val)}
        Use None if you don't want to enforce a specific bound.
        Example: {'age': (0, None), 'score': (None, 100)}
        """
        self.limits = limits

    def fit(self, df: pd.DataFrame):
        return self  # Stateless

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        for col, (min_val, max_val) in self.limits.items():
            if col in df_copy.columns:
                # clip() accepts None for lower/upper to mean "no limit"
                df_copy[col] = df_copy[col].clip(lower=min_val, upper=max_val)
        return df_copy
