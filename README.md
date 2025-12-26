# MLOps Assignment - Full Code & Folder Structure

This repository contains an end-to-end MLOps solution for predicting the risk of heart disease based on the UCI Heart Disease dataset.
The project is designed to be a scalable, reproducible machine learning solution utilizing modern MLOps best practices.

## 🎯 Project Objective

To design, develop, and deploy a scalable, reproducible, and monitored machine learning classifier to predict the presence (1) or absence (0) 
of heart disease based on patient health data, served via a low-latency REST API.


| Aspect | Tool/Technology | Assignment Task |
| :--- | :--- | :--- |
| **Data & Core ML** | Pandas, Scikit-learn, NumPy | 1, 2, 4 |
| **Experiment Tracking** | MLflow | 3 |
| **API Framework** | FastAPI, Uvicorn | 5, 6, 8 |
| **CI/CD Automation** | GitHub Actions, Pytest | 5 |
| **Containerization** | Docker | 6 |
| **Orchestration/Deployment**| Kubernetes (K8s) | 7 |
| **Configuration** | `src/config.py` | 4 (Reproducibility) |

## 🏗️ Architecture Overview

The system architecture decouples the model training pipeline from the high-availability serving API. All stages are automated through configuration and CI/CD.

## 📁 Project Structure

```
mlops-heart-disease/
│
├── data/
│   ├── raw/                    # Original UCI file 
│   │   └── heart.csv
│   ├── processed/              
│   │   └── heart_clean.csv
│   └── download_dataset.py     # Downloads directly from official UCI URL
│
├── notebooks/
│   └── eda.ipynb               # Professional EDA: histograms, correlation heatmap, class balance
│
├── src/
│   ├── preprocess.py           # Data cleaning, imputation, StandardScaler pipeline
│   ├── train.py                # Full training + MLflow logging
│   ├── inference.py            # Single & batch prediction functions
│   ├── config.py               # Paths, hyperparameters, seeds
│   ├── utils.py                # Helper functions (logging, metrics)
│   └── model/
│       └── final_model.pkl     # Best model (RandomForest + scaler in Pipeline)
│
├── api/
│   ├── main.py                 # FastAPI app with /predict, /health, /metrics
│   ├── schema.py               # Pydantic Input schema (13 features)
│   └── Dockerfile              # Production-ready Docker image (multi-stage)
│
├── tests/
│   ├── test_preprocess.py      # Tests data loading, missing values, shape
│   ├── test_train.py           # Tests model performance > 0.85 ROC-AUC
│   └── sample_input.json       # Example request for API testing
│
├── .github/
│   └── ci_pipeline.yml         # Lint → Test → Train → Upload model artifact
│
├── k8s/
│   ├── deployment.yaml         # 3 replicas, rolling update strategy
│   ├── service.yaml            # LoadBalancer service
│   └── prometheus-config.yaml  # Scrape config for /metrics endpoint
│
├── requirements.txt            # All dependencies (pinned versions)
├── REPORT.docx                 # 10-page final report with all screenshots
└── README.md                   # This file
```

## ⚙️ Setup and Local Execution

## Prerequisites

1.  Python 3.10+
2.  `pip`
3.  Docker
4.  `kubectl`

## Installation

### 1️⃣ Clone the repository and install dependencies:

    ```bash
    git clone https://github.com/ashimdas94/MLOps_Assignment_1.git
    cd mlops-heart-disease
    pip install -r requirements.txt
    ```

### 2️⃣ Download dataset
   ```bash
   python -m data.download_dataset
   ```
### 3️⃣ EDA:
   ```bash
   jupyter notebook notebooks/eda.ipynb
   ```
### 4️⃣ Train model:
   ```bash
   python -m src.train
   ```
### 5️⃣ Testing:
   ```bash
   python -m pytest tests/
   ```

### 6️⃣ Run API locally:
   ```bash
   uvicorn src.api.main:app --reload --port 8000
   ```
### 7️⃣ Build Docker image:
   ```bash
   docker build -f api/Dockerfile -t heart-api:latest .
   ```
### 8️⃣ Run via Docker:
   ```bash
   docker run -p 8000:8000 heart-api:latest
   ```


## 🚀 Deployment Using Kubernetes (Minikube)

This project supports local Kubernetes deployment using **Minikube**, simulating a production-like environment with container orchestration, scaling, and monitoring.

### Prerequisites (Additional)

Ensure the following are installed:

* **Minikube**
* **Helm**
* **Docker**
* **kubectl**

Verify installation:

```bash
minikube version
kubectl version --client
helm version
```

### 1️⃣ Start Minikube

```bash
minikube start --driver=docker
```

Configure your shell to use Minikube’s Docker daemon:

```bash
eval $(minikube docker-env)
```

### 2️⃣ Build Docker Image Inside Minikube

```bash
docker build -f api/Dockerfile -t heart-api:latest .
```

### 3️⃣ Deploy Application to Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
```

Verify:

```bash
kubectl get pods
kubectl get svc
```

### 4️⃣ Access the API

**Stable local access (recommended):**

```bash
kubectl port-forward svc/heart-disease-predictor-service 8000:80
```

API available at:

```
http://localhost:8000
```

---

## 📊 Monitoring with Prometheus & Grafana

The API exposes Prometheus-compatible metrics at `/metrics` and is monitored using **Prometheus Operator + Grafana**.

### 1️⃣ Install Prometheus Operator

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
kubectl create namespace monitoring
helm install prometheus prometheus-community/kube-prometheus-stack -n monitoring
```

### 2️⃣ Enable Metrics Scraping

```bash
kubectl apply -f k8s/prometheus-config.yaml
```

Verify targets:

```bash
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9090
```

Open `http://localhost:9090` → **Status → Targets**.

### 3️⃣ Access Grafana

```bash
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80
```

Login:

```
admin / prom-operator
```

### 4️⃣ Useful PromQL Queries

**Prediction request rate:**

```promql
sum(rate(http_request_duration_seconds_count{handler="/predict"}[1m]))
```

**p95 latency:**

```promql
histogram_quantile(
  0.95,
  sum by (le) (
    rate(http_request_duration_seconds_bucket{handler="/predict"}[5m])
  )
)
```

---

## ✅ Summary

* Containerized ML inference service
* Kubernetes-based deployment
* CI/CD-driven training and testing
* MLflow experiment tracking
* Production-grade monitoring with Prometheus & Grafana


