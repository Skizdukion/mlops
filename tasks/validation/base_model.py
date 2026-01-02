from pandas import DataFrame
from typing import Dict
from pandas import DataFrame
from typing import Dict, List, Tuple


class DataFrameValidation:
    ALLOWED_TYPES = {
        "int64",
        "float64",
        "bool",
        "datetime64[ns]",
        "datetime64[us]",
        "object",
        "category",
    }

    def __init__(self, schema: Dict[str, str] = None):
        self.schema = schema
        self._errors: List[str] = []  # Internal storage for errors

    def get_errors(self) -> List[str]:
        """Returns the list of validation errors found during the last process."""
        return self._errors

    def _add_error(self, message: str):
        """Helper to log errors and print them."""
        print(message)
        self._errors.append(message)

    def drop_unexpected_columns(self, data_frame: DataFrame) -> DataFrame:
        if not self.schema:
            return data_frame

        allowed_cols = set(self.schema.keys())
        current_cols = set(data_frame.columns)
        extra_cols = current_cols - allowed_cols

        if extra_cols:
            print(f"Dropping unexpected columns: {extra_cols}")
            data_frame = data_frame.drop(columns=list(extra_cols))

        return data_frame

    def validate_schema(self, data_frame: DataFrame) -> bool:
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

    def validate_no_nulls(self, data_frame: DataFrame) -> bool:
        null_counts = data_frame.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]

        if not cols_with_nulls.empty:
            for col, count in cols_with_nulls.items():
                self._add_error(
                    f"Null Error: Column '{col}' contains {count} null values."
                )
            return False
        return True

    def process_and_validate(self, data_frame: DataFrame) -> Tuple[DataFrame, bool]:
        """
        1. Resets error list
        2. Drops extra columns
        3. Runs validations
        Returns: (Modified DataFrame, Success Boolean)
        """
        self._errors = []  # Clear previous errors before a new run

        df_cleaned = self.drop_unexpected_columns(data_frame)

        # Run validations (using bitwise & ensures both run even if the first fails)
        schema_ok = self.validate_schema(df_cleaned)
        nulls_ok = self.validate_no_nulls(df_cleaned)

        is_valid = schema_ok and nulls_ok

        return df_cleaned, is_valid
