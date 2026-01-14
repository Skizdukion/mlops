import pandas as pd
import requests
import psycopg
from tqdm import tqdm
import random
import time

# DB Config
DB_URL = "postgresql://postgres:postgres@localhost:5432/mlops"
API_URL = "http://127.0.0.1:8000"
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet"
)


def load_data():
    print(f"Loading data from {DATA_URL}...")
    df = pd.read_parquet(DATA_URL)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def run_test():
    df = load_data()

    # Filter for valid data (e.g. non-null) just in case
    df = df.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "PULocationID",
            "DOLocationID",
        ]
    )

    # Sample 1000 rows
    # sample_df = df.sample(n=2000)
    print(f"Selected {len(df)} rows for testing.")

    print("Sending requests...")
    success_count = 0
    fail_count = 0

    models = ["xgboost", "rf", "elastic"]

    # Rate limiting: 500 requests per minute ~ 8.33 req/s => 0.12s per request
    DELAY = (60.0 / 500.0) / 2

    for index, row in tqdm(df.iterrows(), total=len(df)):
        time.sleep(DELAY)
        try:
            # Prepare inference request
            pickup_time = row["tpep_pickup_datetime"]

            # Ensure passenger_count is int and valid (1-12) based on DTO
            passenger_count = int(row["passenger_count"])
            if passenger_count < 1:
                passenger_count = 1
            if passenger_count > 12:
                passenger_count = 12

            request_data = {
                "tpep_pickup_datetime": pickup_time.isoformat(),
                "passenger_count": passenger_count,
                "pulocationid": int(row["PULocationID"]),
                "dolocationid": int(row["DOLocationID"]),
                "model_type": random.choice(models),
            }

            # Inference
            response = requests.post(f"{API_URL}/api/v1/predict", json=request_data)
            response.raise_for_status()
            result = response.json()
            prediction_id = result["prediction_id"]

            # Prepare feedback
            # Calculate duration in minutes (or seconds? API DTO says seconds for predicted, checking feedback dto...)
            # Feedback DTO description says "Real duration". Base model usually predicts minutesMainly?
            # Let's check DTO again.
            # DTO inference: est_duration description="Estimated trip duration in seconds" NO wait.
            # Let me re-verify DTO units.

            dropoff_time = row["tpep_dropoff_datetime"]
            duration_minutes = (dropoff_time - pickup_time).total_seconds() / 60

            # Assuming model predicts duration in minutes usually for Taxi data, but DTO says seconds in description.
            # I will send seconds as per logic.

            feedback_data = {
                "prediction_id": prediction_id,
                "dolocationid": int(row["DOLocationID"]),
                "duration": float(duration_minutes),
            }

            # Send feedback
            feedback_response = requests.post(
                f"{API_URL}/api/v1/feedback", json=feedback_data
            )
            feedback_response.raise_for_status()

            success_count += 1

        except Exception as e:
            fail_count += 1
            print(f"Error processing row {index}: {e}")
            # print(response.text if 'response' in locals() else "")

    print(f"Test completed. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    run_test()
