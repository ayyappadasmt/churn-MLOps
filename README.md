# Churn Prediction — End-to-End MLOps Pipeline

End-to-end MLOps pipeline for predicting customer churn, built to demonstrate
production-oriented machine learning practices including experiment tracking,
hyperparameter optimization, automated testing, containerization, continuous
integration, and model serving.

[![CI](https://github.com/ayyappadasmt/churn-MLOps/actions/workflows/ci.yml/badge.svg)](https://github.com/ayyappadasmt/churn-MLOps/actions/workflows/ci.yml)

## Live Demo

The trained churn prediction model is served through a containerized FastAPI
application and deployed publicly on Render.

- **Live API:** https://churn-mlops-1.onrender.com
- **Interactive Swagger UI:** https://churn-mlops-1.onrender.com/docs

> **Note:** The free Render deployment may take around a minute to wake up
> after a period of inactivity.

---

# MLOps Architecture

```text
                         ┌─────────────────────┐
                         │   Customer Dataset  │
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
                         │  SMOTE Balancing    │
                         │ Training Data Only  │
                         └──────────┬──────────┘
                                    │
                                    ▼
                  ┌──────────────────────────────────┐
                  │     Optuna Hyperparameter        │
                  │          Optimization            │
                  │                                  │
                  │       XGBoost + 20 Trials        │
                  │       Objective: ROC-AUC         │
                  └────────────────┬─────────────────┘
                                   │
                                   ▼
                         ┌─────────────────────┐
                         │  MLflow Tracking    │
                         │                     │
                         │ Parameters         │
                         │ Metrics            │
                         │ Artifacts          │
                         │ Experiment Runs    │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │   Best XGBoost      │
                         │       Model         │
                         └──────────┬──────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     │                             │
                     ▼                             ▼
          ┌─────────────────────┐       ┌─────────────────────┐
          │   Automated Tests   │       │  Model Export /     │
          │       pytest        │       │     Serialization    │
          └──────────┬──────────┘       └──────────┬──────────┘
                     │                             │
                     └──────────────┬──────────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Docker        │
                         │   Container Image   │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │    GitHub Actions   │
                         │         CI          │
                         │                     │
                         │  • Install deps     │
                         │  • Run pytest       │
                         │  • Build Docker     │
                         └──────────┬──────────┘
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │       Render       │
                         │ Cloud Deployment   │
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
                         │    /predict         │
                         │    /health          │
                         │    /docs            │
                         └─────────────────────┘
