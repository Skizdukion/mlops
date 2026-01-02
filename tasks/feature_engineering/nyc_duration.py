import pandas as pd
from tasks.feature_engineering.base_model import FeatureEngineer
from tasks.feature_engineering.feature_transformer import (
    NumericalMedianFiller,
    DatePartTransformer,
    OutlierClipper,
    ValueClipper,
    SkewnessTransformer,
)


class NycDataEngineer(FeatureEngineer):
    def __init__(self):
        transformers = [
            NumericalMedianFiller(colums=["passenger_count"]),
            DatePartTransformer(columns=["tpep_pickup_datetime"]),
            OutlierClipper(columns=["duration"]),
            ValueClipper(limits={"duration": (0, None)}),
            SkewnessTransformer(
                columns=[
                    "passenger_count",
                    "trip_distance",
                    "fare_amount",
                    "extra",
                    "mta_tax",
                    "tip_amount",
                    "tolls_amount",
                    "improvement_surcharge",
                    "total_amount",
                ],
                method="log",
            ),
        ]
        super().__init__(transformers=transformers)

    def fit(self, df: pd.DataFrame):
        """
        Sequentially fits all transformers on the training data.
        Note: We use a copy to avoid side effects during the fit phase.
        """
        temp_df = df.copy()
        temp_df = self.pu_do(temp_df)
        temp_df = self.extract_duration(temp_df)
        return super(temp_df)

    def extract_pu_do(self, df: pd.DataFrame):
        df_copy = df.copy()
        df_copy["pu_do"] = (
            df_copy["pulocationid"].astype(str)
            + "_"
            + df_copy["dolocationid"].astype(str)
        )

        return df_copy

    def extract_duration(self, df: pd.DataFrame):
        df_copy = df.copy()
        df_copy["duration"] = (
            df_copy["tpep_dropoff_datetime"] - df_copy["tpep_pickup_datetime"]
        ).dt.total_seconds() / 60

        return df_copy
