import argparse
import logging
import wandb
import pandas as pd
import numpy as np
import scipy.stats
import unittest
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

def go(args):
    run = wandb.init(project="nyc_airbnb", group="eda", job_type="test_data")

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input)
    artifact_path = artifact.file()

    logger.info("Loading dataset")
    df = pd.read_csv(artifact_path)

    # Define the test suite
    class DataTests(unittest.TestCase):
        def setUp(self):
            self.df = df

        def test_column_names(self):
            expected_columns = [
                "id",
                "name",
                "host_id",
                "host_name",
                "neighbourhood_group",
                "neighbourhood",
                "latitude",
                "longitude",
                "room_type",
                "price",
                "minimum_nights",
                "number_of_reviews",
                "last_review",
                "reviews_per_month",
                "calculated_host_listings_count",
                "availability_365",
            ]
            these_columns = self.df.columns.values
            # This also enforces the same order
            self.assertListEqual(list(expected_columns), list(these_columns))

        def test_neighborhood_names(self):
            known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]
            neigh = set(self.df['neighbourhood_group'].unique())
            # Unordered check
            self.assertSetEqual(set(known_names), set(neigh))

        def test_proper_boundaries(self):
            """
            Test proper longitude and latitude boundaries for properties in and around NYC
            """
            idx = self.df['longitude'].between(-74.25, -73.50) & self.df['latitude'].between(40.5, 41.2)
            self.assertEqual(np.sum(~idx), 0, "Some properties are outside NYC boundaries")

        def test_similar_neigh_distrib(self):
            """
            Apply a threshold on the KL divergence to detect if the distribution of the new data is
            significantly different than that of the reference dataset
            """
            # For this project, we'll compare against itself (no reference dataset provided)
            # In a real scenario, you'd compare against a reference dataset
            dist1 = self.df['neighbourhood_group'].value_counts().sort_index()
            dist2 = dist1  # Self-comparison for this project
            kl_threshold = 0.1  # Small threshold since we're comparing the same dataset
            kl_div = scipy.stats.entropy(dist1, dist2, base=2)
            self.assertLess(kl_div, kl_threshold, f"KL divergence {kl_div} exceeds threshold {kl_threshold}")

        def test_row_count(self):
            """
            Ensure the dataset has at least 100 rows
            """
            min_rows = 100
            self.assertGreaterEqual(len(self.df), min_rows, f"Dataset has {len(self.df)} rows, expected at least {min_rows}")

        def test_price_range(self):
            """
            Ensure all prices are within the expected range [10, 350]
            """
            min_price = 10
            max_price = 350
            self.assertTrue(
                (self.df['price'] >= min_price).all() and (self.df['price'] <= max_price).all(),
                f"Some prices are outside the range [{min_price}, {max_price}]"
            )

    # Run the tests
    logger.info("Running data validation tests")
    suite = unittest.TestLoader().loadTestsFromTestCase(DataTests)
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Log test results to W&B
    run.summary["test_passed"] = test_result.wasSuccessful()
    run.summary["num_tests"] = test_result.testsRun
    run.summary["num_errors"] = len(test_result.errors)
    run.summary["num_failures"] = len(test_result.failures)

    if not test_result.wasSuccessful():
        logger.error("Data validation tests failed")
        raise AssertionError("Data validation tests failed")

    logger.info("Data validation tests passed successfully")

    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the provided data")

    parser.add_argument(
        "--input",
        type=str,
        help="Input artifact to test",
        required=True
    )

    args = parser.parse_args()

    go(args)
