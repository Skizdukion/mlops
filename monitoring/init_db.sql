CREATE TABLE IF NOT EXISTS monitoring_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP WITHOUT TIME ZONE DEFAULT (NOW() AT TIME ZONE 'utc'),
    
    -- Drift Metrics
    prediction_drift FLOAT,
    target_drift FLOAT,
    data_drift FLOAT,
    
    -- Performance Metrics (Regression)
    rmse FLOAT,
    r2 FLOAT,
    mae FLOAT,
    
    -- Drifted Columns Details (optional, storing as JSONB for flexibility)
    drifted_columns JSONB,
    
    -- Metadata
    report_name VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_monitoring_metrics_timestamp ON monitoring_metrics(timestamp);
