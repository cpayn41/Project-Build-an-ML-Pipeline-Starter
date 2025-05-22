import argparse
import wandb
import pytest

def go(args):
    run = wandb.init(project="nyc_airbnb", job_type="data_check")
    run.use_artifact(args.csv)
    run.use_artifact(args.ref)
    pytest.main([
        "test_data.py",
        f"--csv={args.csv}",
        f"--ref={args.ref}",
        f"--kl_threshold={args.kl_threshold}",
        f"--min_price={args.min_price}",
        f"--max_price={args.max_price}"
    ])
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--ref", type=str, required=True)
    parser.add_argument("--kl_threshold", type=float, required=True)
    parser.add_argument("--min_price", type=float, required=True)
    parser.add_argument("--max_price", type=float, required=True)
    args = parser.parse_args()
    go(args)
