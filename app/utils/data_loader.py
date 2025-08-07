import os
import sys
from exception_logging.logger import logging
from exception_logging.exception import CustomException
import pandas as pd


class DataLoader:
    def __init__(self, file_path: str, target_column):
        self.file_path = file_path
        self.target_column = target_column

    def load_data(self):
        try:
            ext = os.path.splitext(self.file_path)[-1].lower()
            if ext !='.csv':
                raise ValueError(f"Unsupported file type: {ext}. Only .csv files are supported.")

            if not os.path.exists(self.file_path):
                raise FileNotFoundError(f"The file {self.file_path} does not exist.")
            data = pd.read_csv(self.file_path)
            logging.info(f"Data loaded successfully from {self.file_path}")

            data.columns = [col.strip().replace(' ', '_') for col in data.columns]
            
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the data.")
            
            
            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]

            return data, X, y, self.target_column
        
        except Exception as e:
            logging.error(f"Error occurred while loading data: {str(e)}")
            raise CustomException(e, sys) from e