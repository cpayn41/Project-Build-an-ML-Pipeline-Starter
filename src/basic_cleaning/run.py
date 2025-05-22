import argparse
import logging
import pandas as pd
import wandb
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    input_artifact = args.input_artifact
    output_artifact = args.output_artifact
    output_type = args.output_type
    output_description = args.output_description
    min_price = args.min_price
    max_price = args.max_price

    logger.info("Initializing W&B run")
    run = wandb.init(project="nyc_airbnb", group="basic_cleaning")

    logger.info("Downloading input artifact: %s", input_artifact)
    try:
        artifact = run.use_artifact(input_artifact)
        artifact_path = artifact.file()
        logger.info("Artifact downloaded to: %s", artifact_path)
    except Exception as e:
        logger.error("Failed to download input artifact: %s", str(e))
        raise

    logger.info("Reading input data")
    try:
        df = pd.read_csv(artifact_path)
        logger.info("Input data shape: %s", df.shape)
    except Exception as e:
        logger.error("Failed to read input data: %s", str(e))
        raise

    logger.info("Cleaning data")
    try:
        df = df[df['price'].between(min_price, max_price)]
        df = df.dropna()
        logger.info("Cleaned data shape: %s", df.shape)
    except Exception as e:
        logger.error("Failed to clean data: %s", str(e))
        raise

    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()

    logger.info("Saving cleaned data")
    try:
        os.makedirs(output_type, exist_ok=True)
        output_path = f"{output_type}/{output_artifact}"
        df.to_csv(output_path, index=False)
        logger.info("Cleaned data saved to: %s", output_path)
    except Exception as e:
        logger.error("Failed to save cleaned data: %s", str(e))
        raise

    logger.info("Logging artifact to W&B")
    try:
        artifact = wandb.Artifact(
            output_artifact,
            type=output_type,
            description=output_description
        )
        artifact.add_file(output_path)
        run.log_artifact(artifact)
        logger.info("Artifact logged successfully: %s", output_artifact)
        # Wait for the artifact to be uploaded
        artifact.wait()
        logger.info("Artifact upload completed: %s", output_artifact)
    except Exception as e:
        logger.error("Failed to log or upload artifact: %s", str(e))
        raise

    logger.info("Finishing W&B run")
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data cleaning step")
    parser.add_argument("--input_artifact", type=str, help="Name and version of the input artifact to clean (e.g., sample.csv:latest)")
    parser.add_argument("--output_artifact", type=str, help="Name of the output cleaned artifact (e.g., clean_sample.csv)")
    parser.add_argument("--output_type", type=str, help="Type of the output artifact (e.g., clean_sample)")
    parser.add_argument("--output_description", type=str, help="Description of the output artifact")
    parser.add_argument("--min_price", type=float, help="Minimum price threshold for filtering data")
    parser.add_argument("--max_price", type=float, help="Maximum price threshold for filtering data")
    args = parser.parse_args()
    go(args)
