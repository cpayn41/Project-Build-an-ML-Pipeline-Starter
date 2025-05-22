#!/usr/bin/env python
"""
This script tests a regression model on a test dataset and logs metrics to W&B.
"""
import argparse
import logging
import pandas as pd
import wandb
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="test_regression_model")
    run.config.update(args)

    # Load the model
    logger.info(f"Fetching model artifact: {args.model_export}")
    model_path = run.use_artifact(args.model_export).download()
    model = mlflow.sklearn.load_model(model_path)

    # Load the test dataset
    logger.info(f"Fetching test artifact: {args.test_artifact}")
    test_path = run.use_artifact(args.test_artifact).file()
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop(columns=["price"])
    y_test = test_df["price"]

    # Evaluate the model
    logger.info("Evaluating model on test data")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    logger.info(f"Test MAE: {mae}")
    logger.info(f"Test R²: {r2}")

    # Log metrics to W&B
    run.summary["test_mae"] = mae
    run.summary["test_r2"] = r2

    # Save results as an artifact
    logger.info(f"Saving test results as {args.output_artifact}")
    with open("test_results.txt", "w") as f:
        f.write(f"Test MAE: {mae}\nTest R²: {r2}\n")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="test_results",
        description="Test results for regression model"
    )
    artifact.add_file("test_results.txt")
    run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a regression model on test data")
    parser.add_argument(
        "--model_export",
        type=str,
        help="Artifact containing the trained model",
        required=True
    )
    parser.add_argument(
        "--test_artifact",
        type=str,
        help="Artifact containing the test dataset",
        required=True
    )
    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output test results",
        required=True
    )
    args = parser.parse_args()
    go(args)
