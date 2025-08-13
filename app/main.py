import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np
import joblib
import os
from app.utils.data_loader import DataLoader
from app.utils.data_cleaning import DataCleaner
from app.utils.feature_engineering import FeatureEngineering  # Assuming you've created this
from app.utils.feature_selection import FeatureSelector  # Assuming you've created this
from app.utils.model_trainer import ModelTrainer
from app.utils.model_evaluator import ModelEvaluator
from app.utils.report_generator import ReportGenerator
from app.config import MODEL_DIR, GCS_BUCKET_NAME

def main(file_path, target_column, features_to_drop, dataset_name):
    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    try:
        # Step 1: Load data
        #file_path = "data/Breast_cancer_dataset.csv"  # or any CSV you're testing with
        #target_column = "diagnosis"     # make sure this exists in the dataset
        loader = DataLoader(file_path, target_column)
        data, X, y, tc = loader.load_data()

        # Step 2: Clean data
        cleaner = DataCleaner(loader)
        data, X_cleaned, y_cleaned, tc  = cleaner.clean_data()

        # Step 3: Drop unwanted features
        #features_to_drop = ['id', 'Unnamed: 32']  # Example — customize as needed
        selector = FeatureSelector(features_to_drop)
        X_selected = selector.drop_features(X_cleaned)

        # Step 4: Feature engineering
        engineer = FeatureEngineering(data_loader=None)
        preprocessor = engineer.create_pipeline(X_selected)

        # Apply the pipeline to the cleaned data
        X_transformed = preprocessor.fit_transform(X_selected)

        X_features = pd.DataFrame(X_transformed)

        if not np.issubdtype(y_cleaned.dtype, np.number):
            le = LabelEncoder()
            y_cleaned = pd.Series(le.fit_transform(y_cleaned), name=target_column)
            logging.info("Target variable encoded using LabelEncoder.")

        trainer = ModelTrainer(X_features, y_cleaned)
        results, X_train, X_test, y_train, y_test = trainer.train_models(dataset_name)


        # Step 5: Evaluate models
        summaries = []
        evaluator = ModelEvaluator(trainer.is_classification)

        for model_name in results.keys():
            # model_path = os.path.join(MODEL_DIR, f"{model_name}_best.pkl")
            # if not os.path.exists(model_path):
            #     logging.warning(f"Model file {model_path} not found, skipping.")
            #     continue

            blob_name = evaluator.get_latest_model_blob_name(GCS_BUCKET_NAME, dataset_name, model_name)
            if not blob_name:
                logging.warning(f"No model found for {model_name} in GCS.")
                continue
            model = evaluator.load_model_from_gcs(GCS_BUCKET_NAME, blob_name)
            summary = evaluator.evaluate_and_save(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test
            )
            summaries.append(summary)

        # Step 6: Generate reports
        """
        report_gen = ReportGenerator()
        html_report = report_gen.generate_html_report(summaries)
        logging.info(f"HTML report generated at: {html_report}")
        pdf_report = report_gen.generate_pdf_report(summaries)
        logging.info(f"PDF report generated at: {pdf_report}")
        """

        # Show preview
        print("✅ Pipeline ran successfully.")
        print("Final features preview:")
        print(X_features.head())
        print("📊 Model training results:")
        for model_name, metrics in results.items():
            print(f"\n🧠 {model_name}:")
            for metric, val in metrics.items():
                print(f"  {metric}: {val}")

        return summaries

    except Exception as e:
        logging.error(f"❌ Pipeline failed: {str(e)}")

# if __name__ == "__main__":
#     main()
