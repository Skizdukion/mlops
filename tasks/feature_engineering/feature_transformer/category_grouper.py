import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class CategoricalGrouper(FeatureTransformer):
    def __init__(
        self, columns: list[str], threshold: float = 0.01, placeholder: str = "Other"
    ):
        self.columns = columns
        self.threshold = threshold
        self.placeholder = placeholder
        self.frequent_categories_ = {}

    def fit(self, df: pd.DataFrame):
        """Identify categories that meet the frequency threshold."""
        for col in self.columns:
            counts = df[col].value_counts(normalize=True)
            # Store only keys that appear more than the threshold
            frequent = counts[counts >= self.threshold].index.tolist()
            self.frequent_categories_[col] = set(frequent)
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Replace rare categories with the placeholder."""
        for col, frequent_set in self.frequent_categories_.items():
            if col in df.columns:
                # If value is not in frequent_set, change it to placeholder
                df[col] = df[col].apply(
                    lambda x: x if x in frequent_set else self.placeholder
                )
        return df
