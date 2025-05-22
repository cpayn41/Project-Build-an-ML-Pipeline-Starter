import pytest
import wandb
import pandas as pd

def pytest_addoption(parser):
    parser.addoption("--csv", action="store", default=None, help="Path to input CSV artifact")
    parser.addoption("--ref", action="store", default=None, help="Path to reference CSV artifact")
    parser.addoption("--kl_threshold", action="store", type=float, default=0.2, help="KL divergence threshold")
    parser.addoption("--min_price", action="store", type=float, default=10.0, help="Minimum price")
    parser.addoption("--max_price", action="store", type=float, default=350.0, help="Maximum price")

@pytest.fixture(scope='session')
def data(request):
    csv_arg = request.config.getoption("--csv")
    if not csv_arg:
        raise ValueError("Must provide --csv option with artifact name (e.g., clean_sample.csv:latest)")
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(csv_arg).file()
    data = pd.read_csv(data_path)
    run.finish()
    return data

@pytest.fixture(scope='session')
def ref_data(request):
    ref_arg = request.config.getoption("--ref")
    if not ref_arg:
        raise ValueError("Must provide --ref option with artifact name (e.g., clean_sample.csv:reference)")
    run = wandb.init(job_type="data_tests", resume=True)
    data_path = run.use_artifact(ref_arg).file()
    data = pd.read_csv(data_path)
    run.finish()
    return data

@pytest.fixture(scope='session')
def kl_threshold(request):
    kl_threshold = request.config.getoption("--kl_threshold")
    if kl_threshold is None:
        raise ValueError("Must provide --kl_threshold option")
    return float(kl_threshold)

@pytest.fixture(scope='session')
def min_price(request):
    min_price = request.config.getoption("--min_price")
    if min_price is None:
        raise ValueError("Must provide --min_price option")
    return float(min_price)

@pytest.fixture(scope='session')
def max_price(request):
    max_price = request.config.getoption("--max_price")
    if max_price is None:
        raise ValueError("Must provide --max_price option")
    return float(max_price)
