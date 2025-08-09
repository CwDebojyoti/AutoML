import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from app.exception_logging.logger import logging
from app.exception_logging.exception import CustomException
from app.utils.data_loader import DataLoader
import sys


class FeatureEngineering:
    def __init__(self, data_loader: DataLoader):
        self.data_loader = data_loader

    def create_pipeline(self, X: pd.DataFrame):
        try:
            logging.info("Creating feature engineering pipeline.")

            # Identify numeric and categorical features
            numeric_cols = X.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

            # Define transformations for numeric features
            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Define transformations for categorical features
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])

            # Combine transformations into a preprocessor
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )

            logging.info("Feature engineering pipeline created successfully.")
            return preprocessor

        except Exception as e:
            logging.error(f"Error occurred while creating feature engineering pipeline: {str(e)}")
            raise CustomException(e, sys) from e
