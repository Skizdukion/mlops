from pandas import DataFrame
from typing import Dict


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

    def drop_unexpected_columns(self, data_frame: DataFrame) -> DataFrame:
        """
        Removes columns from the DataFrame that are not defined in the schema.
        Returns a NEW DataFrame (or modifies the existing one).
        """
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
        """Checks if DataFrame dtypes match the defined schema."""
        if not self.schema:
            return True

        for col, expected_type in self.schema.items():
            if col not in data_frame.columns:
                print(f"Schema Error: Column {col} missing.")
                return False

            actual_type = str(data_frame[col].dtype)
            if actual_type != expected_type:
                print(
                    f"Type Error: Column '{col}' expected {expected_type}, got {actual_type}"
                )
                return False
        return True

    def validate_no_nulls(self, data_frame: DataFrame):
        print("Checking for null values...")
        return not data_frame.isnull().values.any()

    def process_and_validate(self, data_frame: DataFrame) -> tuple[DataFrame, bool]:
        """
        1. Drops extra columns
        2. Runs all validations
        Returns: (Modified DataFrame, Success Boolean)
        """
        # Step 1: Clean the data
        df_cleaned = self.drop_unexpected_columns(data_frame)

        # Step 2: Validate the cleaned data
        is_valid = all(
            [
                self.validate_schema(df_cleaned),
                self.validate_no_nulls(df_cleaned),
            ]
        )

        return df_cleaned, is_valid
