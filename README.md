# NYC Taxi Duration Prediction MLOps Project

This project implements an end-to-end MLOps pipeline for predicting NYC taxi trip duration. It includes:
- **API Gateway**: FastAPI service for real-time predictions.
- **Training Pipeline**: Prefect orchestration for model training and retraining.
- **Monitoring**: Evidently AI for data drift and performance tracking.
- **Infrastructure**: Postgres, MLflow, Grafana.

## Installation

```bash
pip install -r requirements.txt
```

## Project Setup & Startup Guide

### Prerequisites
- Docker & Docker Compose
- Python 3.10+
- Conda (optional but recommended)

### Step-by-Step Startup

1. **Start Infrastructure (Postgres)**
   ```bash
   docker-compose up -d postgres
   ```

2. **Start MLflow Tracking Server**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   *Access at http://localhost:5000*

3. **Start Prefect Server**
   ```bash
   prefect server start
   ```
   *Access at http://localhost:4200 (default)*

4. **Run Database Migrations**
   Ensure database tables are created.
   ```bash
   alembic upgrade head
   ```

5. **Start API Gateway**
   ```bash
   uvicorn api_gateway.app.main:app --reload --host 0.0.0.0 --port 8000
   ```
   *Access Swagger UI at http://localhost:8000/docs*

6. **Start Grafana**
   ```bash
   cd monitoring
   docker-compose up -d
   cd ..
   ```
   *Access at http://localhost:3000 (User: admin, Pass: longpro159)*

7. **Register Prefect Deployment**
   Registers the retraining flow with the Prefect server.
   ```bash
   python scripts/create_deployment.py
   ```

8. **Start Monitoring Workers**
   Run these in separate terminals or background:
   ```bash
   python monitoring/metrics_worker.py
   python monitoring/datadrift_worker.py
   ```

9. **Start Prefect Worker**
   To execute triggered retraining flows:
   ```bash
   prefect worker start --pool 'default-agent-pool'
   ```

## Quick Start
You can use the helper script to run everything (requires `tmux` or multiple terminals, or runs in background).
```bash
./scripts/start_all.sh
```

## Database Migrations (Reference)

- **Create**: `alembic revision --autogenerate -m "msg"`
- **Apply**: `alembic upgrade head`
- **Revert**: `alembic downgrade -1`