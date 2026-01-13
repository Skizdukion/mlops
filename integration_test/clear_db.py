import psycopg

# DB Config
DB_URL = "postgresql://postgres:postgres@localhost:5432/mlops"
API_URL = "http://127.0.0.1:8000"
DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2024-02.parquet"
)


def clear_db():
    print("Clearing database...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("TRUNCATE TABLE nyc_duration_feedback CASCADE")
                cur.execute("TRUNCATE TABLE nyc_duration_inferences CASCADE")
                cur.execute("TRUNCATE TABLE monitoring_metrics CASCADE")
                conn.commit()
        print("Database cleared.")
    except Exception as e:
        print(f"Error clearing database: {e}")


if __name__ == "__main__":
    clear_db()
