import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
import mlflow
from sklearn.metrics import root_mean_squared_error

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("nyc-duration-experiment")

def read_dataframe(filename):
    if filename.endswith(".csv"):
        df = pd.read_csv(filename)

        df.lpep_dropoff_datetime = pd.to_datetime(df.lpep_dropoff_datetime)
        df.lpep_pickup_datetime = pd.to_datetime(df.lpep_pickup_datetime)
    elif filename.endswith(".parquet"):
        df = pd.read_parquet(filename)

    df["duration"] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ["PULocationID", "DOLocationID"]
    df[categorical] = df[categorical].astype(str)

    return df

df_train = read_dataframe('./data/green_tripdata_2021-01.parquet')
df_val = read_dataframe('./data/green_tripdata_2021-02.parquet')

df_train['PU_DO'] = df_train['PULocationID'] + '_' + df_train['DOLocationID']
df_val['PU_DO'] = df_val['PULocationID'] + '_' + df_val['DOLocationID']

categorical = ['PU_DO'] #'PULocationID', 'DOLocationID']
numerical = ['trip_distance']

dv = DictVectorizer()

train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

val_dicts = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dicts)

target = 'duration'
y_train = df_train[target].values
y_val = df_val[target].values

train = xgb.DMatrix(X_train, label=y_train)
valid = xgb.DMatrix(X_val, label=y_val)

best_params = {
    'learning_rate': 0.139209906123887,
    'max_depth': 47,
    'min_child_weight': 1.7321507849861393,
    'objective': 'reg:linear',
    'reg_alpha': 0.03826205780174815,
    'reg_lambda': 0.005154073941829626,
    'seed': 42
}

# mlflow.autolog()
with mlflow.start_run():
    mlflow.set_tag("developer", "luca")
    mlflow.set_tag("model", "xgboost")

    mlflow.log_params(best_params)

    booster = xgb.train(params=best_params, dtrain=train,num_boost_round=1000,evals=[(valid, "validation")], early_stopping_rounds=50)

    y_pred = booster.predict(valid)
    rmse = root_mean_squared_error(y_val, y_pred)
    mlflow.log_metric("rmse", rmse)

# with mlflow.start_run():
#     booster = xgb.train(params=best_params, dtrain=train,num_boost_round=1000,evals=[(valid, "validation")], early_stopping_rounds=50)