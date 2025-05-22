#!/usr/bin/env python
"""
This script splits the provided dataframe in test and remainder
"""
import argparse
import logging
import pandas as pd
import wandb
import tempfile
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="train_val_test_split")
    run.config.update(args)

    logger.info(f"Fetching artifact {args.input_artifact}")
    artifact_local_path = run.use_artifact(args.input_artifact).file()

    df = pd.read_csv(artifact_local_path)

    logger.info("Splitting trainval and test")
    trainval, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_seed,
        stratify=df[args.stratify_by] if args.stratify_by != 'none' else None,
    )

    # Save to output files and log as W&B artifacts
    for df, k in zip([trainval, test], ['trainval', 'test']):
        logger.info(f"Uploading {k}_data.csv dataset")
        with tempfile.NamedTemporaryFile(prefix=f"{k}_data_", suffix=".csv", delete=False) as fp:
            df.to_csv(fp.name, index=False)
            artifact = wandb.Artifact(
                name=f"{k}_data",
                type="dataset",
                description=f"{k} split of dataset"
            )
            artifact.add_file(fp.name)
            run.log_artifact(artifact)

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split test and remainder")
    parser.add_argument("input_artifact", type=str, help="Input artifact to split")
    parser.add_argument(
        "test_size", type=float, help="Size of the test split. Fraction of the dataset, or number of items"
    )
    parser.add_argument(
        "--random_seed", type=int, help="Seed for random number generator", default=42, required=False
    )
    parser.add_argument(
        "--stratify_by", type=str, help="Column to use for stratification", default='none', required=False
    )
    args = parser.parse_args()
    go(args)
