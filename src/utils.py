import mlflow
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def eval_and_log(model, X_train, y_train, X_test, y_test, params, model_type):
    """Train, evaluate, and log results to MLflow."""
    with mlflow.start_run():
        # Train
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Metrics
        rmse = mean_squared_error(y_test, preds, squared=False)
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # Log everything
        mlflow.log_params(params)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, artifact_path="model")

        print(f"[{model_type}] RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")
