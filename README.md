Init

mlflow ui --backend-store-uri sqlite:///mflow.db

sudo nvidia-smi -lgc 900,1200 Set min, max gpu clock
sudo nvidia-smi -rgc remove min max