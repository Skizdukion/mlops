Init

## Start API Server
To start the development server:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db # Start mlflow if not started
uvicorn api_gateway.app.main:app --host 127.0.0.1 --port 8000 --reload
```

prefect server start

python3 -c "from api_gateway.app.main import app; print(app)"


## Database Migrations

### Create a new migration
To generate a new migration file after modifying the models:
```bash
alembic revision --autogenerate -m "Description of changes"
```

### Apply migrations
To apply pending migrations to the database:
```bash
alembic upgrade head
```

### Check migration status
To see the current revision of the database:
```bash
alembic current
```
To show the history of migrations:

### Revert migrations
To undo the last migration:
```bash
alembic downgrade -1
```
To revert to a specific revision:
```bash
alembic downgrade <revision_id>
```
To revert all migrations (empty database):
```bash
alembic downgrade base
```

<!-- NEXT: -->
<!-- 1: Data drift -->
<!-- 1.1: Add col is_current_train to Feedback tbl -->
<!-- 1.2: Update flows/tasks in nyc_taxi_pipeline to add rows on what rows is trained -->
<!-- 1.3: Add seperate workers for monitoring data drift between trained data use and coming data, this worker run on every mintues with the last x new rows -->
<!-- 2: Auto re training if rmse fail behind a thresh hold with new data  -->
<!-- 2.1: Create mutiple metrics ( like last 10k rows ) ( last 1 days ) and last ( 7 days ) -->
<!-- 2.2: Metrics last 10k rows run every 10 minutes, and if the new rows doesnt accumulate enough 10k rows -> pass, 1 days metrics run every 1 days ( but now for testing run on each 15 minutes for test ), and 7 days -->
<!-- 2.3: If a model failing behind a thresh hold in 1 days metrics, start prefect nyc_taxi_pipeline flow/task -->
<!-- 2.4: Get the last 3 days rows data + existing is_current_train row data to get around 150k rows (configurable) to start retrain model -->