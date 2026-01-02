from typing import Type
import pandas as pd
from prefect import flow, task, get_run_logger

# Import your custom modules
from tasks.data_loading import nyc_data_loading
from tasks.evaluation.nyc_duration import NYCDurationEvaluator
from tasks.feature_engineering.nyc_duration import (
    NycCatBoostFeature,
    NycTreeDataFeature,
)
from tasks.model_registry.base_model import MLflowModelRegistry
from tasks.training.base_model import BaseModelTrainer
from tasks.training.nyc_duration import NYCCatBoostTrainer, NYCXGBTrainer
from tasks.validation.base_model import DataFrameValidation
from tasks.validation.constant import NYC_RAW_SCHEMA_VALIDATION


@task(name="Engineering & Label Preparation", retries=1)
def run_feature_engineering(df: pd.DataFrame, model):
    logger = get_run_logger()
    logger.info("Initializing Engineer and processing Features/Labels...")

    if model == "catboost":
        feature_engineer = NycCatBoostFeature()
    else:
        feature_engineer = NycTreeDataFeature()

    # fit_transform creates 'duration', clips it, and OHEs the features
    df_processed = feature_engineer.fit_transform(df)

    return df_processed, feature_engineer


@task(name="Model Training")
def train_model(df_processed: pd.DataFrame, trainer):
    """
    Generic training task that accepts any trainer class inheriting from BaseModelTrainer.
    """
    logger = get_run_logger()
    logger.info(f"Training {trainer.model.__name__}")

    # --- THE CRITICAL STEP: SEPARATE X AND Y ---
    if "duration" not in df_processed.columns:
        raise KeyError("Target column 'duration' missing from processed dataframe.")

    y_train = df_processed["duration"]
    X_train = df_processed.drop(columns=["duration"])

    # Execute the training logic
    trainer.train(X_train, y_train)

    return trainer, X_train, y_train


@task(name="Model Evaluation")
def evaluate_model(trainer, X_test_processed, y_test_real, thresholds):
    logger = get_run_logger()

    # Predict on test set
    y_pred = trainer.predict(X_test_processed)

    # Evaluate
    evaluator = NYCDurationEvaluator(thresholds=thresholds)
    report = evaluator.run_evaluation(y_test_real, y_pred)

    logger.info(f"Evaluation Results: {report['metrics']}")
    return report


@task(name="Validate Raw Data")
def validate_raw_data(df: pd.DataFrame):
    raw_validation = DataFrameValidation(NYC_RAW_SCHEMA_VALIDATION)
    df_cleaned, is_valid = raw_validation.process_and_validate(df)

    if not is_valid:
        errors = raw_validation.get_errors()
        # Join errors into a single string for the exception message
        error_report = "\n- ".join(errors)
        raise ValueError(f"Data Validation Failed:\n- {error_report}")

    return df_cleaned


@flow(name="NYC Taxi Duration Training Pipeline")
def nyc_taxi_pipeline(
    train_urls: list[str],
    test_urls: list[str],
    model_type: str,
):
    train_df = nyc_data_loading(train_urls)

    test_df = nyc_data_loading(test_urls)

    train_df = validate_raw_data(train_df)
    test_df = validate_raw_data(test_df)

    # 3. Process Test Data (Transform only)
    train_df, feature_engineer = run_feature_engineering(train_df, model_type)
    test_df = feature_engineer.transform(test_df)

    if model_type == "catboost":
        trainer = NYCCatBoostTrainer()
    elif model_type == "xgboost":
        trainer = NYCXGBTrainer()
    elif model_type == "rf":
        trainer = NYCXGBTrainer()
    elif model_type == "elastic":
        trainer = NYCXGBTrainer()
    else:
        raise SystemError(f"Unknown Model {model_type}")

    # 4. Train
    trainer, _, _ = train_model(train_df, trainer)

    # 5. Evaluate
    # Extract y_test from the processed test dataframe
    y_test_real = test_df["duration"]
    X_test_final = test_df.drop(columns=["duration"])

    report = evaluate_model(
        trainer, X_test_final, y_test_real, thresholds={"rmse": 10.0}
    )

    # 6. Register with MLflow
    if report["status"] == "PASSED":
        registry = MLflowModelRegistry(experiment_name="NYC-Taxi-Duration")
        registry.register_model(
            model_name="NYC_Duration",
            model=trainer.model,
            eval_report=report,
            engineer=feature_engineer,
        )


if __name__ == "__main__":
    nyc_taxi_pipeline(
        train_urls=[
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet",
        ],
        test_urls=[
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-06.parquet",
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
        ],
        model_type="xgboost",
    )
