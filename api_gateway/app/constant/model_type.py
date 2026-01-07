from enum import Enum

class ModelType(str, Enum):
    catboost = "catboost"
    xgboost = "xgboost"
    rf = "rf"
    elastic = "elastic"
