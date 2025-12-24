import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import json
import base64
import pandas as pd
import tempfile
import os


def prep_df(parquet_urls):
    if isinstance(parquet_urls, str):
        parquet_urls = [parquet_urls]

    # Read and concatenate all Parquet files
    df_list = []
    for url in parquet_urls:
        df_part = pd.read_parquet(url)
        df_part.columns = df_part.columns.str.lower().str.replace(" ", "_")
        df_list.append(df_part)

    df = pd.concat(df_list, ignore_index=True)
    col_to_del = [
        "tpep_pickup_datetime",
        "tpep_dropoff_datetime",
        "ratecodeid",
        "store_and_fwd_flag",
        "payment_type",
        "congestion_surcharge",
        "airport_fee",
        "pulocationid",
        "dolocationid",
    ]

    df["duration"] = (
        df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]
    ).dt.total_seconds() / 60

    dt = df["tpep_pickup_datetime"]
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    df["pu_do"] = df["pulocationid"].astype(str) + "_" + df["dolocationid"].astype(str)

    df["passenger_count"] = df["passenger_count"].fillna(
        df["passenger_count"].mode()[0]
    )

    df = df.drop(columns=col_to_del)

    df = df[
        df["duration"].between(
            0.0001,  # Use a small positive value instead of 0 to be safe
            df["duration"].quantile(0.99),
            inclusive="neither",  # > 0 and < quantile
        )
    ]

    target_col = "duration"

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test

def train_catboost_model(X_train, X_test, y_train, y_test):
    cat_cols = [
        "vendorid",
        "hour",
        "dayofweek",
        "is_weekend",
        "pu_do",
    ]
    cat_feature_indices = [X_train.columns.get_loc(col) for col in cat_cols]

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=8,
        loss_function="RMSE",
        random_seed=42,
        verbose=100,
        task_type="GPU",  # Enable GPU training
        devices="0",  # Use the first GPU
        border_count=32,  # Optimize for GPU speed
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test, y_test),
        use_best_model=True,
    )

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("RMSE:", rmse)
    print("MAE:", mae)
    print("R²:", r2)

    return model


def save_model_to_json(model, filepath='model.json'):
    """
    Save CatBoost model to a JSON file with all necessary components
    """
    # Get model parameters
    params = model.get_params()
    
    # Get categorical feature indices
    cat_indices = model.get_cat_feature_indices()
    
    # Save the model binary data as base64 string
    # First save to temporary file, then encode
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
        tmp_path = tmp.name
    
    # Save model to temporary file
    model.save_model(tmp_path)
    
    # Read binary data and encode as base64
    with open(tmp_path, 'rb') as f:
        model_data = f.read()
    
    # Encode binary data to base64 string
    model_data_b64 = base64.b64encode(model_data).decode('utf-8')
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    # Create JSON structure
    model_json = {
        'model_type': 'CatBoostRegressor',
        'model_data_b64': model_data_b64,
        'parameters': params,
        'categorical_indices': cat_indices,
        'feature_names': model.feature_names_ if hasattr(model, 'feature_names_') else None,
        'metadata': {
            'save_timestamp': pd.Timestamp.now().isoformat(),
            'catboost_version': model.__class__.__module__.split('.')[0]
        }
    }
    
    # Save to JSON file
    with open(filepath, 'w') as f:
        json.dump(model_json, f, indent=2)
    
    print(f"✅ Model saved to {filepath}")
    print(f"   File size: {len(json.dumps(model_json)) / 1024:.2f} KB")
    
    return filepath


def load_model_from_json(filepath='model.json'):
    """
    Load CatBoost model from JSON file
    """
    # Read JSON file
    with open(filepath, 'r') as f:
        model_json = json.load(f)
    
    # Decode base64 model data
    model_data = base64.b64decode(model_json['model_data_b64'])
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(suffix='.cbm', delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(model_data)
    
    # Load model from temporary file
    loaded_model = CatBoostRegressor()
    loaded_model.load_model(tmp_path)
    
    # Clean up temporary file
    os.unlink(tmp_path)
    
    print(f"✅ Model loaded from {filepath}")
    
    # Optional: Print available parameters
    if 'parameters' in model_json:
        print(f"   Model parameters: {list(model_json['parameters'].keys())}")
    
    return loaded_model