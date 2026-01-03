import pandas as pd
from tasks.feature_engineering.feature_transformer.base_model import FeatureTransformer


class FrequencyCatEncoder(FeatureTransformer):
    """
    Encodes categorical columns as integers based on their frequency.

    More frequent categories get lower integer values (1, 2, 3, ...).
    Unknown categories during transform are assigned 0.

    This is memory-efficient and works well with tree-based models.
    """

    def __init__(self, columns: list[str], handle_unknown: str = "zero"):
        """
        :param columns: List of categorical columns to encode
        :param handle_unknown: How to handle unseen categories during transform.
                              "zero" = assign 0, "rare" = assign max_rank + 1
        """
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.encodings_ = {}  # {col_name: {category: rank}}
        self.max_ranks_ = {}  # {col_name: max_rank_value}

    def fit(self, df: pd.DataFrame):
        """Learn the frequency-based encoding for each column."""
        for col in self.columns:
            if col in df.columns:
                # Get value counts (sorted by frequency descending)
                value_counts = df[col].value_counts()

                # Create mapping: most frequent → 1, second → 2, etc.
                encoding_map = {
                    category: rank + 1
                    for rank, category in enumerate(value_counts.index)
                }

                self.encodings_[col] = encoding_map
                self.max_ranks_[col] = len(encoding_map)

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the frequency-based encoding."""
        for col in self.columns:
            if col not in df.columns:
                continue

            if col not in self.encodings_:
                raise ValueError(
                    f"Column '{col}' was not fitted. "
                    f"Available columns: {list(self.encodings_.keys())}"
                )

            encoding_map = self.encodings_[col]

            # Determine value for unknown categories
            if self.handle_unknown == "zero":
                unknown_value = 0
            elif self.handle_unknown == "rare":
                unknown_value = self.max_ranks_[col] + 1
            else:
                raise ValueError(
                    f"Unknown handle_unknown value: {self.handle_unknown}. "
                    f"Expected 'zero' or 'rare'."
                )

            # Map categories to their frequency ranks
            # Use .map() with fillna for unknown categories
            df[col] = df[col].map(encoding_map).fillna(unknown_value).astype(int)

        return df
