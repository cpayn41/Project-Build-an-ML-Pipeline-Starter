import argparse
import logging
import pandas as pd
import wandb
import os

def go(args):
    """
    Clean the input data and produce a cleaned artifact.
    Args:
        input_artifact (str): Name and version of the input artifact to clean (e.g., sample.csv:latest)
        output_artifact (str): Name of the output cleaned artifact (e.g., clean_sample.csv)
        output_type (str): Type of the output artifact (e.g., clean_sample)
        output_description (str): Description of the output artifact
        min_price (float): Minimum price threshold for filtering data
        max_price (float): Maximum price threshold for filtering data
    """
    input_artifact = args.input_artifact
    output_artifact = args.output_artifact
    output_type = args.output_type
    output_description = args.output_description
    min_price = args.min_price
    max_price = args.max_price

    # DO NOT MODIFY
    run = wandb.init(project="nyc_airbnb", group="basic_cleaning")
    artifact = run.use_artifact(input_artifact)
    df = pd.read_csv(artifact.file())
    df = df[df['price'].between(min_price, max_price)]
    os.makedirs(output_type, exist_ok=True)
    output_path = f"{output_type}/{output_artifact}"
    df.to_csv(output_path, index=False)
    artifact = wandb.Artifact(
        output_artifact,
        type=output_type,
        description=output_description
    )
    artifact.add_file(output_path)
    run.log_artifact(artifact)
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data cleaning step")
    parser.add_argument(
	"--input_artifact",
    	type=str,
    	help="Name and version of the input artifact to clean (e.g., sample.csv:latest)"
	)
    parser.add_argument(
        "--output_artifact",
        type=str,
    	help="Name of the output cleaned artifact (e.g., clean_sample.csv)"
    )
    parser.add_argument(
        "--output_type",
        type=str,
    	help="Type of the output artifact (e.g., clean_sample)"
    )
    parser.add_argument(
        "--output_description",
        type=str,
    	help="Description of the output artifact"
    )
    parser.add_argument(
        "--min_price",
        type=float,
    	help="Minimum price threshold for filtering data"
    )
    parser.add_argument(
        "--max_price",
        type=float,
    	help="Maximum price threshold for filtering data"
    )
    args = parser.parse_args()
    go(args)
