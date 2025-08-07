import pandas as pd
from exception_logging.logger import logging
from exception_logging.exception import CustomException
import sys


class FeatureSelector:
    def __init__(self, features_to_drop):
        """
        :param features_to_drop: List of feature names to drop from the dataset.
        """
        self.features_to_drop = features_to_drop

    def drop_features(self, X: pd.DataFrame) -> pd.DataFrame:
        try:
            logging.info(f"Dropping features: {self.features_to_drop}")
            X_dropped = X.drop(columns=self.features_to_drop, errors='ignore')
            logging.info(f"Features dropped successfully. Remaining features: {list(X_dropped.columns)}")
            return X_dropped

        except Exception as e:
            logging.error(f"Error occurred during feature selection: {str(e)}")
            raise CustomException(e, sys)

