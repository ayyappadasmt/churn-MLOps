# End-to-End MLOps Churn Prediction Pipeline

End-to-end MLOps pipeline for predicting customer churn, built to demonstrate production-oriented machine learning practices including experiment tracking, hyperparameter optimization, automated testing, containerization, continuous integration, and model serving.

[![CI](https://github.com/ayyappadasmt/churn-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/ayyappadasmt/churn-MLOps/actions/workflows/ci.yml)

## Live Demo

The trained churn prediction model is served through a containerized FastAPI application and deployed publicly on Render.

- **Live API:** https://churn-mlops-1.onrender.com
- **Interactive Swagger UI:** https://churn-mlops-1.onrender.com/docs

> **Note:** The free Render deployment may take around a minute to wake up after a period of inactivity.
> 
> **Note:** Why Render Was Used Instead of Hugging Face: Hugging Face Spaces was the original deployment target, chosen for its zero-friction Docker-SDK deployment and its recognizability within ML/AI hiring contexts. During deployment, however, Hugging Face's Docker-SDK Spaces were found to require a paid plan rather than being available on the free tier, which was a hard blocker for a zero-budget, time-constrained project.
Render was chosen as the replacement because it satisfied the same requirements without any code changes: it builds directly from an existing Dockerfile, connects natively to a GitHub repository (so pushes can trigger redeploys), offers a genuinely free tier for small web services, and produces a permanent public HTTPS URL suitable for sharing with a recruiter or interviewer. Because the containerization work from Phase 3 was platform-agnostic by design, switching hosts required no 

---

## MLOps Architecture

```text
                    ┌─────────────────────┐
                    │  Customer Dataset   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Data Preprocessing  │
                    │ Cleaning / Encoding │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Train / Test      │
                    │       Split         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   SMOTE Balancing   │
                    │  Training Data Only │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Optuna         │
                    │  Hyperparameter     │
                    │    Optimization     │
                    │                     │
                    │  XGBoost, 20 Trials │
                    │  Objective: ROC-AUC │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  MLflow Tracking    │
                    │                     │
                    │  Parameters         │
                    │  Metrics            │
                    │  Artifacts          │
                    │  Experiment Runs    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Best XGBoost      │
                    │       Model         │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┴─────────────┐
                 ▼                           ▼
      ┌─────────────────────┐    ┌─────────────────────┐
      │  Automated Tests    │    │   Model Export /     │
      │      pytest         │    │    Serialization     │
      └──────────┬──────────┘    └──────────┬──────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Docker        │
                    │   Container Image   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   GitHub Actions    │
                    │         CI          │
                    │                     │
                    │  • Install deps     │
                    │  • Run pytest       │
                    │  • Build Docker     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       Render        │
                    │  Cloud Deployment   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │    REST Service     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  /predict            │
                    │  /health             │
                    │  /docs               │
                    └─────────────────────┘
```

---

## Running It Yourself

### Locally

Clone the repository and install the dependencies:

```bash
git clone https://github.com/ayyappadasmt/churn-MLOps.git
cd churn-MLOps

python -m venv venv
venv\Scripts\activate        # Windows

pip install -r requirements.txt
```

Start the application:

```bash
cd src
python train.py               # Optional — a trained model is already included
uvicorn api.main:app --reload
```

The API will be available at:

```
http://127.0.0.1:8000
```

Open the interactive Swagger API documentation:

```
http://127.0.0.1:8000/docs
```

### With Docker

Build the Docker image:

```bash
docker build -t churn-mlops .
```

Run the container:

```bash
docker run -p 8000:8000 churn-mlops
```

The API will then be available at:

```
http://127.0.0.1:8000
```

Swagger documentation:

```
http://127.0.0.1:8000/docs
```

### Running Tests

Run the complete test suite using:

```bash
pytest tests/ -v
```

This executes the project's automated tests and verifies the core functionality of the application.

### View Experiment History

MLflow is used to track experiments, Optuna trials, parameters, metrics, and model performance.

From the project root:

```bash
cd src
mlflow ui
```

Then open:

```
http://127.0.0.1:5000
```

The MLflow dashboard allows you to compare:

- Optuna hyperparameter optimization trials
- Model parameters
- Evaluation metrics
- Training runs
- The final selected model

---

## Future Additions

This project covers the core MLOps lifecycle end-to-end, but there's a clear next set of practices that would take it further toward a fully production-grade system:

- **Data drift detection** — comparing incoming live request data against the training distribution (e.g., with Evidently AI) to catch when real-world data starts to diverge from what the model was trained on.
- **Automated retraining pipeline** — triggering retraining on a schedule or when drift/performance degradation is detected, rather than only ever manually.
- **Live prediction monitoring** — logging every prediction request/response, and once true outcomes become available, reconciling them against predictions to track real-world accuracy decay over time.
- **Model registry with staged promotion** — using MLflow's Model Registry to move models through `staging → production` with explicit approval gates, instead of always deploying the latest trained artifact.
- **Alerting** — wiring failed CI runs, deployment errors, or detected drift into Slack/email notifications instead of relying on manually checking dashboards.
- **A/B testing or shadow deployment** — running a challenger model alongside the production model on live traffic before fully promoting it.
- **Infrastructure as code** — managing the deployment configuration (Render service, environment variables, scaling settings) declaratively instead of through a manual dashboard setup.
- **Feature store** — centralizing feature computation so training and serving pull from the same consistent source, reducing the risk of training-serving skew as the project grows.
