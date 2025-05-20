# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from geopy.distance import geodesic
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO)

def preprocess_data(input_path, output_path):
    """
    Preprocess the Airbnb dataset and save the cleaned data.
    
    Args:
        input_path (str): Path to the raw dataset (e.g., sample.csv).
        output_path (str): Path to save the preprocessed dataset.
    """
    # Load the data
    logging.info("Loading data from %s", input_path)
    df = pd.read_csv(input_path)

    # Remove rows with missing price or critical categorical features
    logging.info("Removing rows with missing price or critical categorical features")
    df.dropna(subset=['price'], inplace=True)
    df.dropna(subset=['neighbourhood_group', 'room_type'], inplace=True)
    # Note: Missing values in name, host_name, last_review, and reviews_per_month
    # are deferred to the inference pipeline for production consistency

    # Convert last_review to datetime and extract features
    logging.info("Converting last_review to datetime")
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    # Extract year and month, using 0 as a placeholder for missing values
    df['last_review_year'] = df['last_review'].dt.year.fillna(0).astype(int)
    df['last_review_month'] = df['last_review'].dt.month.fillna(0).astype(int)
    # Drop the original last_review column (missing values remain as NaT)
    df.drop(columns=['last_review'], inplace=True)

    # Filter price outliers based on stakeholder input
    logging.info("Filtering price outliers to range $10-$350")
    df = df[(df['price'] >= 10) & (df['price'] <= 350)]

    # Handle other outliers
    logging.info("Handling outliers in minimum_nights and calculated_host_listings_count")
    # Cap minimum_nights at 30 days (unrealistic values like 1250 nights are outliers)
    df['minimum_nights'] = df['minimum_nights'].clip(upper=30)
    # Cap calculated_host_listings_count at 50 (extreme values like 327 are outliers)
    df['calculated_host_listings_count'] = df['calculated_host_listings_count'].clip(upper=50)

    # Handle rare categories in neighbourhood
    logging.info("Grouping rare neighbourhoods into 'Other'")
    neighbourhood_counts = df['neighbourhood'].value_counts()
    rare_neighbourhoods = neighbourhood_counts[neighbourhood_counts < 50].index
    df['neighbourhood'] = df['neighbourhood'].replace(rare_neighbourhoods, 'Other')

    # Log-transform skewed numerical features
    logging.info("Log-transforming skewed numerical features")
    for col in ['minimum_nights', 'number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count']:
        df[col] = np.log1p(df[col])

    # Derive geographical feature: distance to Times Square
    logging.info("Deriving distance to Times Square")
    times_square = (40.7580, -73.9855)  # Times Square coordinates
    df['distance_to_times_square'] = df.apply(
        lambda row: geodesic((row['latitude'], row['longitude']), times_square).miles, axis=1
    )

    # Define features for modeling
    numerical_features = ['latitude', 'longitude', 'minimum_nights', 'number_of_reviews', 
                         'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
                         'last_review_year', 'last_review_month', 'distance_to_times_square']
    categorical_features = ['neighbourhood_group', 'room_type', 'neighbourhood']

    # Create preprocessing pipeline
    logging.info("Creating preprocessing pipeline")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
    )

    # Apply preprocessing
    logging.info("Applying preprocessing")
    X = df[numerical_features + categorical_features]
    y = np.log1p(df['price'])
    X_transformed = preprocessor.fit_transform(X)

    # Create a new DataFrame with transformed features
    cat_columns = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    transformed_columns = numerical_features + list(cat_columns)
    X_transformed_df = pd.DataFrame(X_transformed, columns=transformed_columns)

    # Save the preprocessed data
    X_transformed_df['price'] = y.reset_index(drop=True)
    logging.info("Saving preprocessed data to %s", output_path)
    X_transformed_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    preprocess_data(sys.argv[1], sys.argv[2])
