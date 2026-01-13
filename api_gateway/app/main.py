# import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api_gateway.app.services.load_model import mlflow_model_management_service
from fastapi import Request
from fastapi.responses import JSONResponse
from api_gateway.app.routes import feedback, inference


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


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred"},
    )


app.include_router(inference.router, prefix="/api/v1", tags=["inference"])
app.include_router(feedback.router, prefix="/api/v1", tags=["feedback"])


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api_gateway.app.main:app", host="127.0.0.1", port=8000, reload=True)
