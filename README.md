\# Churn Prediction — MLOps Pipeline



End-to-end pipeline for predicting customer churn, built to demonstrate

real MLOps practices: experiment tracking, hyperparameter optimization,

automated testing, containerization, and model serving.



\## Live Demo

\[Coming in Phase 6 — will add deployed URL here]



\## Architecture

Data → Preprocessing → SMOTE balancing → Optuna hyperparameter search 

→ MLflow tracking → FastAPI serving → Docker → CI/CD (GitHub Actions) → Deployment



\## Tech Stack

\- \*\*Model\*\*: XGBoost, tuned via Optuna (20-trial search, optimizing ROC-AUC)

\- \*\*Experiment tracking\*\*: MLflow (nested runs per trial)

\- \*\*Imbalance handling\*\*: SMOTE on training data only

\- \*\*Serving\*\*: FastAPI with Pydantic input validation

\- \*\*Testing\*\*: pytest

\- \*\*Containerization\*\*: Docker

\- \*\*CI/CD\*\*: GitHub Actions (coming in Phase 5)



\## Run locally

\\```bash

git clone https://github.com/YOUR-USERNAME/churn-mlops.git

cd churn-mlops

python -m venv venv

venv\\Scripts\\activate

pip install -r requirements.txt

cd src

python train.py

uvicorn api.main:app --reload

\\```

Then visit http://127.0.0.1:8000/docs



\## Run with Docker

\\```bash

docker build -t churn-mlops .

docker run -p 8000:8000 churn-mlops

\\```

