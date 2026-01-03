from pandas import DataFrame
from typing import Dict
from pandas import DataFrame
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class DataFrameValidation:
    ALLOWED_TYPES = {
        "int32",
        "int64",
        "float32",
        "float64",
        "bool",
        "datetime64[ns]",
        "datetime64[us]",
        "object",
        "category",
    }

    def __init__(self, schema: Dict[str, str] = None):
        self.schema = schema
        self._errors: List[str] = []

    def get_errors(self) -> List[str]:
        """Returns the list of validation errors found during the last process."""
        return self._errors

    def _add_error(self, message: str):
        """Helper to log errors."""
        print(message)
        self._errors.append(message)

    def drop_unexpected_columns(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """Removes columns not defined in the schema."""
        if not self.schema:
            return data_frame

        allowed_cols = set(self.schema.keys())
        current_cols = set(data_frame.columns)
        extra_cols = current_cols - allowed_cols

        if extra_cols:
            print(f"Dropping unexpected columns: {extra_cols}")
            data_frame = data_frame.drop(columns=list(extra_cols))

        return data_frame

    def coerce_types(self, data_frame: pd.DataFrame) -> pd.DataFrame:
        """
        Attempts to convert columns to the types specified in the schema.
        This solves the int32 vs int64 issue by forcing the 'expected' type.
        IMPORTANT: This method modifies the input DataFrame in-place.
        """
        if not self.schema:
            return data_frame

        for col, expected_type in self.schema.items():
            if col in data_frame.columns:
                try:
                    # Only convert if types actually differ to save performance
                    if str(data_frame[col].dtype) != expected_type:
                        data_frame[col] = data_frame[col].astype(expected_type)
                except (ValueError, TypeError) as e:
                    self._add_error(
                        f"Coercion Error: Could not convert '{col}' to {expected_type}: {e}"
                    )

        return data_frame

    def validate_schema(self, data_frame: pd.DataFrame) -> bool:
        """Checks if columns exist and match the required types."""
        if not self.schema:
            return True

        valid = True
        for col, expected_type in self.schema.items():
            if col not in data_frame.columns:
                self._add_error(f"Schema Error: Column '{col}' missing.")
                valid = False
                continue

            actual_type = str(data_frame[col].dtype)
            if actual_type != expected_type:
                self._add_error(
                    f"Type Error: Column '{col}' expected {expected_type}, got {actual_type}"
                )
                valid = False
        return valid

    def validate_no_nulls(self, data_frame: pd.DataFrame) -> bool:
        """Checks for missing values in the DataFrame."""
        null_counts = data_frame.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if not cols_with_nulls.empty:
            for col, count in cols_with_nulls.items():
                self._add_error(
                    f"Null Error: Column '{col}' contains {count} null values."
                )
            return False
        return True

    def process_and_validate(
        self, data_frame: pd.DataFrame
    ) -> Tuple[pd.DataFrame, bool]:
        """
        1. Resets error list
        2. Drops extra columns
        3. Coerces types (e.g., int32 -> int64)
        4. Runs final validations
        Returns: (Modified DataFrame, Success Boolean)
        """
        self._errors = []

        # 1. Clean structure
        df_processed = self.drop_unexpected_columns(data_frame)

        # 2. Fix types (Solves your int32/int64 issue)
        df_processed = self.coerce_types(df_processed)

        # 3. Final Check
        schema_ok = self.validate_schema(df_processed)
        nulls_ok = self.validate_no_nulls(df_processed)

        is_valid = schema_ok and nulls_ok

        return df_processed, is_valid
