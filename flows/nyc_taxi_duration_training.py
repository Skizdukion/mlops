import pandas as pd
import optuna

from prefect import flow, task, get_run_logger
from tasks.evaluation.nyc_duration import NYCDurationEvaluator
from tasks.tuning.nyc_duration import HyperparameterTuner
from tasks.feature_engineering.nyc_duration import (
    NycCatBoostFeature,
    NycTreeDataFeature,
)
from tasks.model_registry.base_model import MLflowModelRegistry
from tasks.training.nyc_duration import (
    NYCCatBoostTrainer,
    NYCXGBTrainer,
    NYCRandomForestTrainer,
    NYCElasticNetTrainer,
)
from tasks.validation.base_model import DataFrameValidation
from tasks.validation.constant import (
    NYC_RAW_SCHEMA_VALIDATION,
    NYC_SCHEMA_VALIDATION_FOR_CATBOOST,
    NYC_SCHEMA_VALIDATION_FOR_TREE,
)
from tasks.data_loading import nyc_data_loading
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


from optuna.trial import TrialState


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
def train_model(df_processed: pd.DataFrame, trainer, **params):
    """
    Generic training task that accepts any trainer class inheriting from BaseModelTrainer.
    """
    logger = get_run_logger()
    logger.info(f"Training {trainer.model.__class__.__name__}")

    # --- THE CRITICAL STEP: SEPARATE X AND Y ---
    if "duration" not in df_processed.columns:
        raise KeyError("Target column 'duration' missing from processed dataframe.")

    y_train = df_processed["duration"]
    X_train = df_processed.drop(columns=["duration"])

    # Execute the training logic
    trainer.train(X_train, y_train, **params)

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
    df_cleaned = raw_validation.drop_unexpected_columns(df)
    df_processed = raw_validation.coerce_types(df_cleaned)
    schema_ok = raw_validation.validate_schema(df_processed)

    if not schema_ok:
        errors = raw_validation.get_errors()
        # Join errors into a single string for the exception message
        error_report = "\n- ".join(errors)
        raise ValueError(f"Data Validation Failed:\n- {error_report}")

    return df_processed


@task(name="Validate Processed Data")
def validate_processed_data(df: pd.DataFrame, model_type: str):
    # Use model-specific schema validation
    if model_type == "catboost":
        schema = NYC_SCHEMA_VALIDATION_FOR_CATBOOST
    else:
        schema = NYC_SCHEMA_VALIDATION_FOR_TREE

    raw_validation = DataFrameValidation(schema)
    df_cleaned, is_valid = raw_validation.process_and_validate(df)

    if not is_valid:
        errors = raw_validation.get_errors()
        # Join errors into a single string for the exception message
        error_report = "\n- ".join(errors)
        raise ValueError(f"Data Validation Failed:\n- {error_report}")

    return df_cleaned


@task(name="Load data")
def load_train_test_data(train_urls, test_urls):
    train_df = nyc_data_loading(train_urls)
    test_df = nyc_data_loading(test_urls)

    return train_df, test_df


@task(name="Hyper params tuning")
def hyper_params_tuning(df, trainer_class):
    logger = get_run_logger()

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    X_train = train_df.drop(columns=["duration"])
    X_val = val_df.drop(columns=["duration"])
    y_train = train_df["duration"].values
    y_val = val_df["duration"].values

    params_tuner = HyperparameterTuner(
        trainer_class, X_train, y_train, X_val, y_val, mean_squared_error
    )
    # Enable Optuna logging to show progress in terminal
    optuna.logging.set_verbosity(optuna.logging.INFO)

    study = params_tuner.optimize()

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger.info("Study statistics: ")
    logger.info(f"Number of finished trials: {len(study.trials)}")
    logger.info(f"Number of pruned trials: {len(pruned_trials)}")
    logger.info(f"Number of complete trials: {len(complete_trials)}")

    return study.best_params


@flow(name="NYC Taxi Duration Training Pipeline")
def nyc_taxi_pipeline(
    train_urls: list[str],
    test_urls: list[str],
    model_type: str,
):
    train_df, test_df = load_train_test_data(train_urls, test_urls)

    train_df = validate_raw_data(train_df)
    test_df = validate_raw_data(test_df)

    # 3. Process Test Data (Transform only)
    train_df, feature_engineer = run_feature_engineering(train_df, model_type)
    test_df = feature_engineer.transform(test_df)

    train_df = validate_processed_data(train_df, model_type)
    test_df = validate_processed_data(test_df, model_type)

    if model_type == "catboost":
        trainer_class = NYCCatBoostTrainer().__class__
    elif model_type == "xgboost":
        trainer_class = NYCXGBTrainer().__class__
    elif model_type == "rf":
        trainer_class = NYCRandomForestTrainer().__class__
    elif model_type == "elastic":
        trainer_class = NYCElasticNetTrainer().__class__
    else:
        raise SystemError(f"Unknown Model {model_type}")

    # 4. Hyper params tuning
    best_params = hyper_params_tuning(train_df, trainer_class)

    trainer = trainer_class(model_params=best_params)

    trainer_class, _, _ = train_model(train_df, trainer)
    # 5. Evaluate
    # Extract y_test from the processed test dataframe
    y_test_real = test_df["duration"]
    X_test_final = test_df.drop(columns=["duration"])

    report = evaluate_model(
        trainer_class, X_test_final, y_test_real, thresholds={"rmse": 10.0}
    )

    # 6. Register with MLflow
    if report["status"] == "PASSED":
        registry = MLflowModelRegistry(experiment_name="NYC-Taxi-Duration")
        model_name = f"NYC_Duration_{model_type}"
        version = registry.register_model(
            model_name=model_name,
            model=trainer_class.model,
            eval_report=report,
            feature_engineering=feature_engineer,
        )
        registry.promote_to_production(model_name, version, "rmse")


if __name__ == "__main__":

    train_urls = [
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2022-01.parquet",
    ]

    test_urls = [
        # "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-06.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-02.parquet",
    ]

    # nyc_taxi_pipeline(
    #     train_urls=train_urls,
    #     test_urls=test_urls,
    #     model_type="catboost",
    # )

    nyc_taxi_pipeline(
        train_urls=train_urls,
        test_urls=test_urls,
        model_type="xgboost",
    )

    # nyc_taxi_pipeline(
    #     train_urls=train_urls,
    #     test_urls=test_urls,
    #     model_type="rf",
    # )

    # nyc_taxi_pipeline(
    #     train_urls=train_urls,
    #     test_urls=test_urls,
    #     model_type="elastic",
    # )
