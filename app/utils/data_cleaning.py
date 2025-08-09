import os
import sys
from app.exception_logging.logger import logging
from app.exception_logging.exception import CustomException
import pandas as pd
from app.utils.data_loader import DataLoader

class DataCleaner:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def clean_data(self):
        try:
            data, X, y, target_column = self.data_loader.load_data()
            logging.info("Starting data cleaning process.")

            # Identify numeric and categorical features
            numeric_cols = X.select_dtypes(include=['number']).columns
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns

            # Fill missing numeric values with median
            for col in numeric_cols:
                if X[col].isnull().sum() > 0:
                    median = X[col].median()
                    X[col].fillna(median, inplace=True)
                    logging.info(f"Filled missing values in numeric column '{col}' with median: {median}")

            # Fill missing categorical values with mode
            for col in categorical_cols:
                if X[col].isnull().sum() > 0:
                    mode = X[col].mode()[0]
                    X[col].fillna(mode, inplace=True)
                    logging.info(f"Filled missing values in categorical column '{col}' with mode: {mode}")

            # Strip and lower-case categorical values
            for col in categorical_cols:
                X[col] = X[col].astype(str).str.strip().str.lower()
            
            logging.info("Data cleaning completed successfully.")
            return data, X, y, target_column
        
        except Exception as e:
            logging.error(f"Error occurred during data cleaning: {str(e)}")
            raise CustomException(e, sys) from e