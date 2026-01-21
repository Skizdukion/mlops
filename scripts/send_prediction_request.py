import requests
from tqdm import tqdm
import random
import time
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from tasks.data_loading import nyc_data_loading

# DB Config
DB_URL = "postgresql://postgres:postgres@localhost:5432/mlops"
API_URL = "http://127.0.0.1:8000"


from concurrent.futures import ThreadPoolExecutor, as_completed


def process_row(index, row, models, delay):
    time.sleep(delay)
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
            "pulocationid": int(row["pulocationid"]),
            "dolocationid": int(row["dolocationid"]),
            "model_type": random.choice(models),
        }

        # Inference
        response = requests.post(f"{API_URL}/api/v1/predict", json=request_data)
        response.raise_for_status()
        result = response.json()
        prediction_id = result["prediction_id"]

        # Prepare feedback
        dropoff_time = row["tpep_dropoff_datetime"]
        duration_minutes = (dropoff_time - pickup_time).total_seconds() / 60

        feedback_data = {
            "prediction_id": prediction_id,
            "dolocationid": int(row["dolocationid"]),
            "duration": float(duration_minutes),
        }

        # Send feedback
        feedback_response = requests.post(
            f"{API_URL}/api/v1/feedback", json=feedback_data
        )
        feedback_response.raise_for_status()

        return True, None

    except Exception as e:
        return False, f"Error processing row {index}: {e}"


def run_test():
    # df = load_data()
    df = nyc_data_loading(
        [
            "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-03.parquet"
        ]
    )

    # Filter for valid data (e.g. non-null) just in case
    df = df.dropna(
        subset=[
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "passenger_count",
            "pulocationid",
            "dolocationid",
        ]
    )

    # Sample 1000 rows
    # sample_df = df.sample(n=2000)
    print(f"Selected {len(df)} rows for testing.")

    print("Sending requests...")
    success_count = 0
    fail_count = 0

    models = ["xgboost", "rf", "elastic"]

    # Rate limiting: 1500 requests per minute
    # We are running 3 threads. To maintain ~1500 req/min total, each thread should handle ~500 req/min?
    # Or strict global rate limit?
    # The previous code had DELAY = (60.0 / 1500.0) / 2 = 0.02s per request.
    # If we want it "faster" we can just let it rip or reduce sleep.
    # But user asked for "3 parallel requests", implying concurrency.
    # If we keep same delay per thread, we get 3x throughput.
    DELAY = 0

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for index, row in df.iterrows():
            futures.append(executor.submit(process_row, index, row, models, DELAY))

        for future in tqdm(as_completed(futures), total=len(futures)):
            success, error = future.result()
            if success:
                success_count += 1
            else:
                fail_count += 1
                print(error)

    print(f"Test completed. Success: {success_count}, Failed: {fail_count}")


if __name__ == "__main__":
    run_test()
