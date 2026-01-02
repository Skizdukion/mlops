import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class OneHotEncoderTransformer(FeatureTransformer):
    def __init__(self, columns: list[str], handle_unknown: str = "ignore"):
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.categories_ = {}  # To store the learned categories for each column

    def fit(self, df: pd.DataFrame):
        """Learn all unique categories for the specified columns."""
        for col in self.columns:
            if col in df.columns:
                # Store the unique categories as a list
                self.categories_[col] = df[col].dropna().unique().tolist()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply one-hot encoding based on the categories learned during fit."""
        df_copy = df.copy()

        for col, cats in self.categories_.items():
            for cat in cats:
                new_col_name = f"{col}_{cat}"
                # Create a binary column: 1 if matches category, else 0
                df_copy[new_col_name] = (df_copy[col] == cat).astype(int)

            # Drop the original categorical column
            df_copy = df_copy.drop(columns=[col])

        return df_copy
