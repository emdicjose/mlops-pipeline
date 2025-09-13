import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# set tracking uri
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# set model name
model_name = "ElasticNet"

# set experiment name 
mlflow.set_experiment("diabetes-example")

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random hyperparameters
alpha = np.random.uniform(0.01, 1.0)
l1_ratio = np.random.uniform(0.0, 1.0)

with mlflow.start_run():
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    rmse = mean_squared_error(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    # Log params, metrics, model
    mlflow.log_param("alpha", alpha)
    mlflow.log_param("l1_ratio", l1_ratio)
    mlflow.log_param("model_type", "ElasticNet")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    # Infer model signature
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.sklearn.log_model(
        model,
        name=model_name,
        signature=signature,
        input_example=X_train[:5]
    )
    print(f"[ElasticNet] RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
