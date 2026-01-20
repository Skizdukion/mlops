#!/bin/bash
session="mlops_project"
conda_act="source /home/lpk/miniconda3/bin/activate shadowing"

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "tmux could not be found. Please install tmux to use this script."
    echo "sudo apt install tmux"
    exit 1
fi

echo "Starting MLOps Project in tmux session: $session"

# Start New Session
tmux new-session -d -s $session

# Window 1: Infrastructure (Postgres & Grafana & Migrations)
tmux rename-window -t $session:0 'infra'
tmux send-keys -t $session:0 "$conda_act" C-m
tmux send-keys -t $session:0 'docker-compose up -d postgres' C-m
tmux send-keys -t $session:0 'cd monitoring && docker-compose up -d && cd ..' C-m
tmux send-keys -t $session:0 'sleep 5 && alembic upgrade head' C-m
tmux send-keys -t $session:0 'echo "Infra started. Run scripts/create_deployment.py once server is ready."' C-m

# Window 2: MLFlow
tmux new-window -t $session:1 -n 'mlflow'
tmux send-keys -t $session:1 "$conda_act" C-m
tmux send-keys -t $session:1 'mlflow ui --backend-store-uri sqlite:///mlflow.db' C-m

# Window 3: Prefect Server
tmux new-window -t $session:2 -n 'prefect-server'
tmux send-keys -t $session:2 "$conda_act" C-m
tmux send-keys -t $session:2 'prefect server start' C-m

# Window 4: API Gateway
tmux new-window -t $session:3 -n 'api'
tmux send-keys -t $session:3 "$conda_act" C-m
tmux send-keys -t $session:3 'uvicorn api_gateway.app.main:app --host 0.0.0.0 --port 8000' C-m

# Window 5: Prefect Worker
tmux new-window -t $session:4 -n 'worker-pool'
tmux send-keys -t $session:4 "$conda_act" C-m
tmux send-keys -t $session:4 'echo "Waiting for prefect server..." && sleep 10' C-m
tmux send-keys -t $session:4 'prefect worker start --pool default-agent-pool' C-m

# Window 6: Monitoring Workers
tmux new-window -t $session:5 -n 'monitors'
tmux send-keys -t $session:5 "$conda_act" C-m
tmux split-window -h -t $session:5
tmux send-keys -t $session:5.1 "$conda_act" C-m
tmux send-keys -t $session:5.0 'python monitoring/metrics_worker.py' C-m
tmux send-keys -t $session:5.1 'python monitoring/datadrift_worker.py' C-m

# Window 7: Deployment Registration (Auto)
tmux new-window -t $session:6 -n 'deploy'
tmux send-keys -t $session:6 "$conda_act" C-m
tmux send-keys -t $session:6 'echo "Waiting for prefect server..." && sleep 15' C-m
tmux send-keys -t $session:6 'python scripts/create_deployment.py' C-m

# Select first window
tmux select-window -t $session:0

echo "---------------------------------------------------------"
echo "🚀 MLOps Project Started successfully in tmux session!"
echo "---------------------------------------------------------"
echo "Service              | URL / Command"
echo "---------------------|-----------------------------------"
echo "MLFlow UI            | http://localhost:5000"
echo "Prefect Server       | http://localhost:4200"
echo "API Gateway          | http://localhost:8000/docs"
echo "Grafana              | http://localhost:3000 (admin/longpro159)"
echo "---------------------------------------------------------"
echo "Attach to session    | tmux attach -t $session"
echo "Attach to Log Stream | tmux attach -t $session:infra"
echo "Attach to API        | tmux attach -t $session:api"
echo "Attach to Monitors   | tmux attach -t $session:monitors"
echo "List sessions        | tmux ls"
echo "Kill session         | tmux kill-session -t $session"
echo "---------------------------------------------------------"
