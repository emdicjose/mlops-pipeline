import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

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

# Define search space
space = {
    "alpha": hp.uniform("alpha", 0.01, 1.0),
    "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0)
}

# Objective function
def objective(params):
    with mlflow.start_run(
        run_name = f"{model_name}_alpha={params['alpha']:.2f}_l1_ratio={params['l1_ratio']:.2f}"
    ):
        model = ElasticNet(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # Log params, metrics, model
        mlflow.log_params(params)
        mlflow.log_metric("rmse", rmse)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(model, name="model", signature=signature, input_example=X_train[:5])

    return {"loss": rmse, "status": STATUS_OK}

trials = Trials()

best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=5,  # number of hyperparameter combinations to try
    trials=trials
)

print("Best hyperparameters:", best)
