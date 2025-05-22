import pytest
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

def test_column_names(data):
    expected_columns = [
        'id', 'name', 'host_id', 'host_name', 'neighbourhood_group',
        'neighbourhood', 'latitude', 'longitude', 'room_type', 'price',
        'minimum_nights', 'number_of_reviews', 'last_review',
        'reviews_per_month', 'calculated_host_listings_count',
        'availability_365'
    ]
    assert set(data.columns) == set(expected_columns), "Column names do not match expected columns"

def test_neighborhood_names(data, ref_data):
    expected_neighborhoods = set(ref_data['neighbourhood'].unique())
    current_neighborhoods = set(data['neighbourhood'].unique())
    assert current_neighborhoods.issubset(expected_neighborhoods), "New neighborhood names detected"

def test_proper_boundaries(data):
    assert data['latitude'].between(40.5, 41.2).all(), "Latitude out of bounds"
    assert data['longitude'].between(-74.3, -73.7).all(), "Longitude out of bounds"

def test_similar_neigh_distrib(data, ref_data, kl_threshold, request):
    threshold = request.config.getoption("--kl_threshold")
    data_neigh = data['neighbourhood'].value_counts(normalize=True).sort_index()
    ref_neigh = ref_data['neighbourhood'].value_counts(normalize=True).sort_index()
    stat, _ = ks_2samp(data_neigh, ref_neigh)
    assert stat < threshold, f"Neighborhood distribution differs too much (KS stat: {stat}, threshold: {threshold})"

def test_row_count(data):
    assert 15000 < data.shape[0] < 1000000, "Row count out of expected range"

def test_price_range(data, request):
    min_price = request.config.getoption("--min_price")
    max_price = request.config.getoption("--max_price")
    assert data['price'].between(min_price, max_price).all(), f"Prices out of range [{min_price}, {max_price}]"
