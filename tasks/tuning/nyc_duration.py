import optuna
import pandas as pd
from typing import Dict, Any, Type
from tasks.training.base_model import BaseModelTrainer
from tasks.training.nyc_duration import (
    NYCCatBoostTrainer,
    NYCXGBTrainer,
    NYCRandomForestTrainer,
    NYCElasticNetTrainer,
)


class HyperparameterTuner:
    def __init__(
        self,
        model_class: Type[BaseModelTrainer],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        metric_func,
    ):
        self.model_class = model_class
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.metric_func = metric_func

    def objective(self, trial):
        params = self.suggest_params(trial)

        # Initialize trainer with trial params
        trainer = self.model_class(model_params=params)

        # Train
        trainer.train(self.X_train, self.y_train)

        # Validate
        preds = trainer.predict(self.X_val)
        score = self.metric_func(self.y_val, preds)

        return score

    def suggest_params(self, trial) -> Dict[str, Any]:
        # Define search spaces based on model type
        if issubclass(self.model_class, NYCCatBoostTrainer):
            return {
                "iterations": trial.suggest_int("iterations", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 1.0, log=True
                ),
                "depth": trial.suggest_int("depth", 4, 10),
                "l2_leaf_reg": trial.suggest_float(
                    "l2_leaf_reg", 1e-8, 100.0, log=True
                ),
                "loss_function": "RMSE",
                "verbose": 0,
            }
        elif issubclass(self.model_class, NYCXGBTrainer):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-3, 1.0, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "n_jobs": -1,
            }
        elif issubclass(self.model_class, NYCRandomForestTrainer):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "n_jobs": -1,
            }
        elif issubclass(self.model_class, NYCElasticNetTrainer):
            return {
                "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
                "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
                "max_iter": 1000,
            }
        else:
            return {}

    def optimize(self, n_trials=10, direction="minimize"):
        study = optuna.create_study(direction=direction)
        study.optimize(self.objective, n_trials=n_trials)

        return study
