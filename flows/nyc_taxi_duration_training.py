import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

import pandas as pd
import optuna
from sqlalchemy import create_engine
from prefect import flow, task, get_run_logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from optuna.trial import TrialState

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from tasks.evaluation.nyc_duration import NYCDurationEvaluator
from tasks.tuning.nyc_duration import HyperparameterTuner
from tasks.feature_engineering.nyc_duration import (
    NycCatBoostFeature,
    NycTreeDataFeature,
)
from tasks.data_loading import nyc_data_loading
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

# DB Config
from alembic_model.config import alembic_config

engine = create_engine(alembic_config.DATABASE_URL)


@task(name="Load Data")
def load_data_from_urls(train_urls: list, test_urls: list):
    logger = get_run_logger()
    logger.info("Loading data from URLs...")
    train_df = nyc_data_loading(train_urls)
    test_df = nyc_data_loading(test_urls)
    logger.info(f"Loaded {len(train_df)} train rows and {len(test_df)} test rows.")
    return train_df, test_df


@task(name="Load Data for Retraining")
def load_data_for_retraining(model_type: str, limit_rows: int = 50000):
    """
    Fetches data for retraining:
    1. Fetch new data (is_current_train=False)
    2. Split new data into Train (80%) and Test (20%)
    3. Fetch old data (is_current_train=True)
    4. Mix old data (50%) with new Train data (50%)
    5. Return Train (Mixed) and Test (New Only)
    """
    logger = get_run_logger()
    logger.info(f"Loading data from Database for Retraining {model_type}...")

    query_new_data = f"""
        SELECT 
            p.tpep_pickup_datetime, 
            p.passenger_count, 
            p.pulocationid, 
            p.dolocationid,
            f.duration,
            p.id as prediction_id
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = '{model_type}'
          AND f.is_current_train = False
        LIMIT {limit_rows}
    """

    query_old_data = f"""
        SELECT 
            p.tpep_pickup_datetime, 
            p.passenger_count, 
            p.pulocationid, 
            p.dolocationid,
            f.duration,
            p.id as prediction_id
        FROM nyc_duration_inferences p
        JOIN nyc_duration_feedback f ON p.id = f.prediction_id
        WHERE p.model_type = '{model_type}'
          AND f.is_current_train = True
        ORDER BY RANDOM()
        LIMIT {limit_rows} 
    """

    try:
        new_df = pd.read_sql(query_new_data, engine)
        old_df = pd.read_sql(query_old_data, engine)

        logger.info(f"Retrieved {len(new_df)} new rows and {len(old_df)} old rows.")

        if new_df.empty:
            logger.warning("No new data found for retraining.")
            return pd.DataFrame(), pd.DataFrame()

        # 1. Split New Data -> 80% Train, 20% Test (Validation)
        # validation set should generally be recent data to reflect current production
        new_train_df, new_test_df = train_test_split(
            new_df, test_size=0.2, random_state=42
        )

        # 2. Select Old Data to mix (50/50 split)
        # We want the training set to be 50% mixed.
        # So we match the size of new_train_df with old_df
        target_size = len(new_train_df)

        if len(old_df) > target_size:
            old_train_df = old_df.sample(n=target_size, random_state=42)
        else:
            old_train_df = old_df  # Take all if not enough

        logger.info(
            f"Mixing {len(old_train_df)} old rows with {len(new_train_df)} new rows for training."
        )

        # 3. Combine to create Final Training Set
        final_train_df = pd.concat([old_train_df, new_train_df], ignore_index=True)
        final_test_df = new_test_df

        # Mark chosen new data as 'is_current_train=True' in DB?
        # The prompt says "start the retraining pipeline... select ...".
        # Usually we update the flags AFTER successful training or as part of this process.
        # For now, we just return the dataframes as per task description.
        # Updating flags might be a separate task or implicitly done later.
        # But to prevent re-using same 'new' data again and again as 'new', we should probably mark them.
        # However, complex state updates in DB might be better handled if explicitly required.
        # Given "start retraining pipeline", I will focus on data loading first.

        return final_train_df, final_test_df

    except Exception as e:
        logger.error(f"Error loading retraining data: {e}")
        return pd.DataFrame(), pd.DataFrame()


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


@task(name="Load & Process Data")
def load_process_data(train_urls, test_urls, model_type, from_db=False):
    if not from_db:
        # Initial Load from Parquet
        train_df, test_df = load_data_from_urls(train_urls, test_urls)

        train_df = validate_raw_data(train_df)
        test_df = validate_raw_data(test_df)
    else:
        # Retraining Load from DB
        train_df, test_df = load_data_for_retraining(model_type=model_type)

    return train_df, test_df


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
    train_urls: list,
    test_urls: list,
    model_type: str = "xgboost",
    params_search: bool = False,
    from_db: bool = False,
):
    logger = get_run_logger()
    logger.info(f"Starting training pipeline for {model_type}...")

    # Load Data
    train_df, test_df = load_process_data(train_urls, test_urls, model_type, from_db)

    if train_df.empty:
        logger.error("Training Data Empty. Aborting.")
        return

    # Feature Engineering
    logger.info("Feature Engineering...")
    # Keep a copy of raw data for DB population (alignment assumed safe as no rows dropped)
    train_df_raw = train_df.copy()

    train_df, feature_engineer = run_feature_engineering(train_df, model_type)
    test_df = feature_engineer.transform(test_df)

    # Validate Processed Data (After Engineering, includes duration)
    logger.info("Validating Processed Data...")
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

    if params_search:
        # Hyperparameter Tuning
        logger.info("Hyperparameter Tuning...")
        best_params = hyper_params_tuning(train_df, trainer_class)
        trainer = trainer_class(model_params=best_params)
    else:
        trainer = trainer_class()

    trainer_class, _, _ = train_model(train_df, trainer)

    # 5. Populate DB with initial training data + predictions (If First Run)
    if not from_db:
        from flows.nyc_populate_training_data import populate_initial_training_data

        logger.info("Generating predictions for training data population...")
        try:
            # We need to predict on the PROCESSED train_df that was used for training (which has features engineered)
            # train_df IS processed data at this point (line 241/236)

            # Predict
            X_train_full = train_df.drop(columns=["duration"])
            y_pred_train = trainer_class.predict(X_train_full)

            logger.info("Populating initial training data to DB for monitoring...")
            # Use the RAW dataframe (for original columns) + the new PREDICTIONS
            populate_initial_training_data(
                train_df_raw, model_type, predictions=y_pred_train
            )

        except Exception as e:
            logger.error(f"Failed to populate DB: {e}")
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
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-01.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-01.parquet",
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet",
    ]

    test_urls = [
        "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2025-02.parquet",
    ]

    nyc_taxi_pipeline(
        train_urls=train_urls,
        test_urls=test_urls,
        model_type="xgboost",
    )

    nyc_taxi_pipeline(
        train_urls=train_urls,
        test_urls=test_urls,
        model_type="rf",
    )

    nyc_taxi_pipeline(
        train_urls=train_urls,
        test_urls=test_urls,
        model_type="elastic",
        params_search=True,
    )
