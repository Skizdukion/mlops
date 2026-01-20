from pathlib import Path
import sys

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from flows.nyc_taxi_duration_training import nyc_taxi_pipeline


from prefect.client.orchestration import get_client
from prefect.exceptions import ObjectNotFound
import asyncio
from prefect.client.schemas.actions import WorkPoolCreate


async def get_or_create_work_pool(pool_name: str):
    async with get_client() as client:
        try:
            pool = await client.read_work_pool(work_pool_name=pool_name)
            print(f"Work pool '{pool_name}' already exists: {pool.id}")
        except ObjectNotFound:
            print(f"Work pool '{pool_name}' not found. Creating...")
            # Create a process type work pool
            wp = WorkPoolCreate(name=pool_name, type="process")
            await client.create_work_pool(work_pool=wp)
            print(f"Work pool '{pool_name}' created successfully.")


async def deployment_exists(flow_name: str, deployment_name: str) -> bool:
    async with get_client() as client:
        try:
            # We assume flow name matches file/function name or is predictable.
            # Actually client.read_deployment_by_name expects "flow-name/deployment-name"
            # Our flow name in @flow decorator is "NYC Taxi Duration Training Pipeline"
            full_name = f"{flow_name}/{deployment_name}"
            await client.read_deployment_by_name(full_name)
            return True
        except ObjectNotFound:
            return False


def deploy():
    # Ensure work pool exists
    pool_name = "default-agent-pool"
    try:
        asyncio.run(get_or_create_work_pool(pool_name))
    except Exception as e:
        print(
            f"Warning: automatic work pool creation failed ({e}). Proceeding to deploy assumes it exists."
        )

    flow_name = "NYC Taxi Duration Training Pipeline"
    deployment_name = "regular-training"

    # Check if deployment exists to avoid re-registering every time (optional optimization)
    # But often re-registering is good to update code changes.
    # User asked "dont register if the deployment already exist".
    try:
        exists = asyncio.run(deployment_exists(flow_name, deployment_name))
        if exists:
            print(
                f"Deployment '{flow_name}/{deployment_name}' already exists. Skipping registration."
            )
            return
    except Exception as e:
        print(
            f"Warning: could not check existing deployment ({e}). Proceeding to register."
        )

    # ... Deploy Logic ...
    nyc_taxi_pipeline.from_source(
        source=str(project_root),
        entrypoint="flows/nyc_taxi_duration_training.py:nyc_taxi_pipeline",
    ).deploy(
        name=deployment_name,
        work_pool_name=pool_name,
        parameters={
            "model_type": "xgboost",
            "from_db": True,
            "train_urls": [],
            "test_urls": [],
        },
        cron=None,
        build=False,
        push=False,
    )

    print(f"Deployment '{deployment_name}' successfully created!")


if __name__ == "__main__":
    deploy()
