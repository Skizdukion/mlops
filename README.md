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