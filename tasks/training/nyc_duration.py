from catboost import CatBoostRegressor
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from xgboost import XGBRegressor
from tasks.training.base_model import BaseModelTrainer

class NYCXGBTrainer(BaseModelTrainer):
    def init_model(self) -> XGBRegressor:
        # Default params if none provided
        params = self.model_params or {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
        }
        return XGBRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        self.logger.info("Fitting XGBoost model...")
        # You can pass eval_set via kwargs in your orchestration task
        self.model.fit(X_train, y_train, **kwargs)

class NYCCatBoostTrainer(BaseModelTrainer):
    def init_model(self) -> CatBoostRegressor:
        params = self.model_params or {
            "iterations": 300,
            "learning_rate": 0.05,
            "depth": 8,
            "loss_function": "RMSE",
            "random_seed": 42,
            "verbose": 100,
            "task_type": "GPU",  # Enable GPU training
            "devices": "0",  # Use the first GPU
            "border_count": 32,  # Optimize for GPU speed
        }
        return CatBoostRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        self.logger.info("Fitting Catboost model...")
        # You can pass eval_set via kwargs in your orchestration task
        self.model.fit(X_train, y_train, **kwargs)

class NYCRandomForestTrainer(BaseModelTrainer):
    def init_model(self) -> RandomForestRegressor:
        # Default params optimized for a balance of speed and accuracy
        params = self.model_params or {
            "n_estimators": 100,
            "max_depth": 12,  # Prevents the trees from growing too deep (overfitting)
            "min_samples_split": 5,
            "random_state": 42,
            "n_jobs": -1,  # Uses all available CPU cores
        }
        return RandomForestRegressor(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        self.logger.info("Fitting Random Forest model...")
        self.model.fit(X_train, y_train)


class NYCElasticNetTrainer(BaseModelTrainer):
    def init_model(self) -> ElasticNet:
        # l1_ratio=0.5 means an equal mix of Lasso and Ridge
        params = self.model_params or {
            "alpha": 1.0,  # Constant that multiplies the penalty terms
            "l1_ratio": 0.5,  # The ElasticNet mixing parameter
            "max_iter": 2000,  # Increased to ensure convergence
            "random_state": 42,
        }
        return ElasticNet(**params)

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, **kwargs):
        self.logger.info("Fitting ElasticNet model...")
        self.model.fit(X_train, y_train)
