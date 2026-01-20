import psycopg

# DB Config
DB_URL = "postgresql://postgres:postgres@localhost:5432/mlops"


def remove_untraining_data():
    print("Removing non-training data (is_current_train=False)...")
    try:
        with psycopg.connect(DB_URL) as conn:
            with conn.cursor() as cur:
                # 0. Optmization: Create indexes to speed up the query if they don't exist
                # This might take a moment but saves huge time on scans.
                print("Checking/Creating indexes...")
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_feedback_is_current_train ON nyc_duration_feedback (is_current_train);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_feedback_prediction_id ON nyc_duration_feedback (prediction_id);"
                )
                conn.commit()
                print("Indexes ready.")

                batch_size = 10000
                total_deleted = 0

                while True:
                    # 1. Delete a batch from feedback calling RETURNING to get the prediction_ids
                    # We use a subquery with LIMIT for batching

                    batch_query = f"""
                            WITH deleted_batch AS (
                                DELETE FROM nyc_duration_feedback
                                WHERE id IN (
                                    SELECT id 
                                    FROM nyc_duration_feedback 
                                    WHERE is_current_train = False 
                                    LIMIT {batch_size}
                                )
                                RETURNING prediction_id
                            )
                            DELETE FROM nyc_duration_inferences
                            WHERE id IN (SELECT prediction_id FROM deleted_batch);
                        """

                    cur.execute(batch_query)
                    rows_affected = cur.rowcount
                    conn.commit()

                    total_deleted += rows_affected
                    print(
                        f"Batch deleted: {rows_affected} records. Total so far: {total_deleted}"
                    )

                    if rows_affected == 0:
                        break

                cur.execute("TRUNCATE TABLE monitoring_metrics CASCADE")

        print(f"Non-training data removed. Total: {total_deleted}")
    except Exception as e:
        print(f"Error removing data: {e}")


if __name__ == "__main__":
    remove_untraining_data()
