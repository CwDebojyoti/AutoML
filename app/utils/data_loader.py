import os
import sys
from app.exception_logging.logger import logging
from app.exception_logging.exception import CustomException
from google.cloud import storage
import pandas as pd


class DataLoader:
    def __init__(self, file_source, target_column):
        self.file_source = file_source
        self.target_column = target_column

    def upload_files_to_gcs(self, files, bucket_name, blob_name):
    
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_file(files, content_type="text/csv")

        return f"gs://{bucket_name}/{blob_name}"


    def load_data(self):
        try:
            # If file_source is a string, treat as file path
            if isinstance(self.file_source, str):
                self.file_source.seek(0)
                ext = os.path.splitext(self.file_source)[-1].lower()
                if ext != '.csv':
                    raise ValueError(f"Unsupported file type: {ext}. Only .csv files are supported.")
                if not os.path.exists(self.file_source):
                    raise FileNotFoundError(f"The file {self.file_source} does not exist.")
                data = pd.read_csv(self.file_source)
                logging.info(f"Data loaded successfully from {self.file_source}")
            else:
                # Assume file-like object (BytesIO)
                self.file_source.seek(0)  # Reset pointer to start
                data = pd.read_csv(self.file_source)
                logging.info("Data loaded successfully from file-like object (e.g., GCS BytesIO)")

            data.columns = [col.strip().replace(' ', '_') for col in data.columns]
            if self.target_column not in data.columns:
                raise ValueError(f"Target column '{self.target_column}' not found in the data.")

            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]

            return data, X, y, self.target_column

        except Exception as e:
            logging.error(f"Error occurred while loading data: {str(e)}")
            raise CustomException(e, sys) from e