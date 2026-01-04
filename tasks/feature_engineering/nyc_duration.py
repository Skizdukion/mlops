import pandas as pd
from tasks.feature_engineering.base_model import FeatureEngineer
from tasks.feature_engineering.feature_transformer import (
    NumericalMedianFiller,
    DatePartTransformer,
    OutlierClipper,
    ValueClipper,
    SkewnessTransformer,
    FrequencyCatEncoder,
    OneHotEncoderTransformer,
)


class BaseNycFeature(FeatureEngineer):
    def _apply_domain_logic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Shared NYC logic used by ALL models"""
        if "pulocationid" in df.columns and "dolocationid" in df.columns:
            df["pu_do"] = (
                df["pulocationid"].astype(str) + "_" + df["dolocationid"].astype(str)
            )

        # 2. Target Engineering (Only runs if dropoff time is present - i.e., Training)
        # In Production, we won't have the dropoff time yet!
        if (
            "tpep_dropoff_datetime" in df.columns
            and "tpep_pickup_datetime" in df.columns
        ):
            df["duration"] = (
                df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
            ).dt.total_seconds() / 60
        else:
            # Optional: Log that we are in 'Inference Mode'
            pass

        return df

    def fit(self, df: pd.DataFrame):
        return super().fit(self._apply_domain_logic(df))

    def transform(self, df: pd.DataFrame):
        return super().transform(self._apply_domain_logic(df))


class NycTreeDataFeature(BaseNycFeature):
    def __init__(self):
        transformers = [
            NumericalMedianFiller(columns=["passenger_count"]),
            DatePartTransformer(columns=["tpep_pickup_datetime"]),
            ValueClipper(limits={"duration": (0.1, None)}),
            OutlierClipper(
                columns=["duration"],
                lower_percentile=0.01,
                upper_percentile=0.95,
            ),
            SkewnessTransformer(
                columns=[
                    "passenger_count",
                ],
                method="log",
            ),
            # Use FrequencyCatEncoder instead of OneHot to avoid 291GB memory usage
            FrequencyCatEncoder(columns=["pu_do"]),
            OneHotEncoderTransformer(
                columns=[
                    "tpep_pickup_datetime_hour",
                    "tpep_pickup_datetime_month",
                    "tpep_pickup_datetime_year",
                    "tpep_pickup_datetime_weekday",
                    # "tpep_pickup_datetime_is_weekend",
                ]
            ),
        ]
        super().__init__(transformers=transformers)


class NycCatBoostFeature(BaseNycFeature):
    def __init__(self):
        transformers = [
            NumericalMedianFiller(columns=["passenger_count"]),
            DatePartTransformer(columns=["tpep_pickup_datetime"]),
            ValueClipper(limits={"duration": (0.1, None)}),
            OutlierClipper(
                columns=["duration"],
                lower_percentile=0.01,
                upper_percentile=0.95,
            ),
            SkewnessTransformer(
                columns=[
                    "passenger_count",
                ],
                method="log",
            ),
        ]
        super().__init__(transformers=transformers)
