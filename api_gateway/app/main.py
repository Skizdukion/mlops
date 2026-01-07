# import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_gateway.app.services.load_model import mlflow_model_management_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown."""
    print("Starting MLflow Model Management...")
    mlflow_model_management_service.start()

    yield

    print("Shutting down MLflow Model Management...")
    mlflow_model_management_service.stop()


app = FastAPI(
    title="Taxi Duration Prediction API",
    description="API Gateway for taxi duration prediction system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}
