Init

mlflow ui --backend-store-uri sqlite:///mlflow.db

prefect server start

python3 -c "from api_gateway.app.main import app; print(app)"