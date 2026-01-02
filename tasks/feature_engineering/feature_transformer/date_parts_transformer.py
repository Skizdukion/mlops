import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class DatePartTransformer(FeatureTransformer):
    def __init__(self, columns: list[str], features: list[str] = None):
        """
        :param columns: List of columns to transform.
        :param features: List of date parts to extract (e.g., ['day', 'month', 'is_weekend']).
        """
        self.columns = columns
        self.features = features or ["year", "month", "day", "weekday"]

    def fit(self, df: pd.DataFrame):

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()

        for col in self.columns:
            if col not in df_copy.columns:
                raise ValueError(f"Column {col} not found in DataFrame")

            # Ensure it's a datetime object
            if not pd.api.types.is_datetime64_any_dtype(df_copy[col]):
                df_copy[col] = pd.to_datetime(df_copy[col])

            # Apply requested transformations
            if "year" in self.features:
                df_copy[f"{col}_year"] = df_copy[col].dt.year
            if "month" in self.features:
                df_copy[f"{col}_month"] = df_copy[col].dt.month
            if "weekday" in self.features:
                df_copy[f"{col}_weekday"] = df_copy[col].dt.weekday
            if "is_weekend" in self.features:
                df_copy[f"{col}_is_weekend"] = (
                    df_copy[col].dt.weekday.isin([5, 6]).astype(int)
                )

        return df_copy
