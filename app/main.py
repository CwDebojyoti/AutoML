import logging
import pandas as pd
from utils.data_loader import DataLoader
from utils.data_cleaning import DataCleaner
from utils.feature_engineering import FeatureEngineering  # Assuming you've created this
from utils.feature_selection import FeatureSelector  # Assuming you've created this

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Step 1: Load data
        file_path = "data/healthcare_dataset.csv"  # or any CSV you're testing with
        target_column = "Test_Results"     # make sure this exists in the dataset
        loader = DataLoader(file_path, target_column)
        data, X, y, tc = loader.load_data()

        # Step 2: Clean data
        cleaner = DataCleaner(loader)
        data, X_cleaned, y_cleaned, tc  = cleaner.clean_data()

        # Step 3: Drop unwanted features
        features_to_drop = ["Name", "Doctor", "Hospital", "Insurance_Provider"]  # Example — customize as needed
        selector = FeatureSelector(features_to_drop)
        X_selected = selector.drop_features(X_cleaned)

        # Step 4: Feature engineering
        engineer = FeatureEngineering(data_loader=None)
        preprocessor = engineer.create_pipeline(X_selected)

        # Apply the pipeline to the cleaned data
        X_transformed = preprocessor.fit_transform(X_selected)

        X_features = pd.DataFrame(X_transformed)

        # Show preview
        print("✅ Pipeline ran successfully.")
        print("Final features preview:")
        print(X_features.head())

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")

if __name__ == "__main__":
    main()
