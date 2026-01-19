# Diabetes Prediction API (Kubernetes Deployment)

A machine learning API that predicts diabetes risk using a trained logistic regression model, built with FastAPI, logs predictions to PostgreSQL, and is deployed on Kubernetes.

---

## Quick Start

### Deploy the API on Kubernetes

#### 1. **Build the API image locally**:
   ```bash
   docker build -t diabetes-api:local ./api


#### 2. Deploy the database:

kubectl apply -f k8s/database-deployment.yaml
kubectl apply -f k8s/database-service.yaml


#### 3. Wait for the database pod to be ready:

kubectl get pods -l app=diabetes-db


#### 4. Deploy the API:

kubectl apply -f k8s/api-deployment.yaml
kubectl apply -f k8s/api-service.yaml


#### 5. Check running resources:

kubectl get deployments
kubectl get pods
kubectl get services

### Use the API

Get the service URL using:

kubectl get services diabetes-api-service

Then visit the interactive documentation at:

http://localhost:8000/docs for interactive FastAPI documentation.

#### Example Prediction:
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "pregnancies": 2,
  "glucose": 130,
  "blood_pressure": 80,
  "skin_thickness": 20,
  "insulin": 85,
  "bmi": 28.5,
  "diabetes_pedigree_function": 0.5,
  "age": 35
}'

#### Response
{
  "prediction": 1,
  "probability": 0.87,
  "risk_level": "High"
}


All predictions are automatically logged to PostgreSQL.

## Model Performance

- Algorithm: Logistic Regression
- Training Dataset: Pima Indians Diabetes Dataset
- Accuracy: ~77–80% on test data
- Target: Binary diabetes classification (0 = No, 1 = Yes)

## API Endpoints
Method	Endpoint	Description
GET	/	Health check
POST	/predict	Single diabetes prediction
GET	/model-info	Model metadata

## Input Parameters
| Parameter                     | Type  | Description                          | Example |
|------------------------------|-------|--------------------------------------|---------|
| `pregnancies`                | int   | Number of pregnancies                | `2`     |
| `glucose`                    | float | Plasma glucose concentration         | `130`   |
| `blood_pressure`             | float | Diastolic blood pressure             | `80`    |
| `skin_thickness`             | float | Triceps skin fold thickness          | `20`    |
| `insulin`                    | float | 2-hour serum insulin                 | `85`    |
| `bmi`                        | float | Body mass index                      | `28.5`  |
| `diabetes_pedigree_function` | float | Diabetes pedigree function           | `0.5`   |
| `age`                        | int   | Age in years                         | `35`    |

## Database Logging

Predictions are stored in PostgreSQL with the following schema:

CREATE TABLE IF NOT EXISTS prediction_logs (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    input_data JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    probability FLOAT NOT NULL,
    risk_level VARCHAR(10) NOT NULL
);

## Development
Local API Development (without Kubernetes)
cd api
pip install -r requirements.txt
python app.py


The API will run at:

http://localhost:8000

## Kubernetes Architecture

- API Deployment: 2 replicas for high availability
- Database Deployment: PostgreSQL 15
- Database Service: ClusterIP (internal access only)
- API Service: LoadBalancer (external access)
- Networking: Kubernetes DNS (diabetes-db-service)

Notes
emptyDir volume is used for learning/demo purposes

For production:

Use PersistentVolumeClaims (PVCs)

Store credentials in Kubernetes Secrets

Push images to a container registry

Add liveness/readiness probes

License

This project is for educational purposes only.