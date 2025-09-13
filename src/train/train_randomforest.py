import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

from hyperopt import fmin, tpe, hp, Trials, STATUS_OK

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# set tracking uri
os.environ['MLFLOW_TRACKING_URI'] = "http://127.0.0.1:5000"
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# set model name
model_name = "rf"
model_type = "RandomForest"


mlflow.set_experiment("diabetes-example")

# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define search space
space = {
    "n_estimators": hp.quniform("n_estimators", 50, 200, 10),
    "max_depth": hp.quniform("max_depth", 2, 10, 1),
    "max_features": hp.choice("max_features", ["sqrt", "log2"]),
    "min_samples_split": hp.quniform("min_samples_split", 2, 5, 1),
    "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 3, 1)
}

# Objective function
def objective(params):
    with mlflow.start_run(
        run_name = f"{model_name}_n{params['n_estimators']:.2f}_d{params['max_depth']:.2f}_mf_{params['max_features']}"
        ):
        
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            max_features=params['max_features'],
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            random_state=42)
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log params
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)

        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # Log model
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
