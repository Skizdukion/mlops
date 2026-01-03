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
        """Apply one-hot encoding using an optimized concatenation approach."""
        all_new_columns = []

        for col, cats in self.categories_.items():
            if col not in df.columns:
                continue

            # Generate all binary columns for this categorical feature at once
            for cat in cats:
                new_col_name = f"{col}_{cat}"
                # Create a Series and give it the correct name
                new_series = (df[col] == cat).astype(int).rename(new_col_name)
                all_new_columns.append(new_series)

        # 1. Combine all the new binary columns into one DataFrame
        if all_new_columns:
            df_new_features = pd.concat(all_new_columns, axis=1)
            # 2. Join the new features back to the original (excluding processed columns)
            df = pd.concat(
                [df.drop(columns=self.categories_.keys()), df_new_features], axis=1
            )
        else:
            # If no new columns were created, just drop the original columns
            df.drop(columns=self.categories_.keys(), errors="ignore", inplace=True)

        return df
