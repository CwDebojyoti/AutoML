import sys
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from exception_logging.exception import CustomException
from config import (
    CLASSIFIERS, REGRESSORS, MODEL_PARAMS,
    TEST_SIZE, RANDOM_STATE, CROSS_VAL_FOLDS
)


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.is_classification = self._determine_task_type()

    def _determine_task_type(self):
        """Decide whether the task is classification or regression based on target values."""
        if pd.api.types.is_numeric_dtype(self.y):
            unique_values = self.y.nunique()
            if unique_values <= 10:
                logging.info(f"Detected classification task (target has {unique_values} unique values).")
                return True
            else:
                logging.info("Detected regression task.")
                return False
        else:
            logging.info("Detected classification task (non-numeric target).")
            return True

    def train_models(self):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE
            )

            models = CLASSIFIERS if self.is_classification else REGRESSORS
            results = {}

            for name, model in models.items():
                logging.info(f"Training model: {name}")
                
                # Apply custom params if available
                if name in MODEL_PARAMS:
                    model.set_params(**MODEL_PARAMS[name])
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Classification metrics
                if self.is_classification:
                    acc = accuracy_score(y_test, y_pred)
                    cv = cross_val_score(model, self.X, self.y, cv=CROSS_VAL_FOLDS).mean()
                    logging.info(f"{name} Accuracy: {acc:.4f}, CV Score: {cv:.4f}")
                    results[name] = {
                        "accuracy": acc,
                        "cv_score": cv,
                        "report": classification_report(y_test, y_pred, output_dict=True)
                    }
                # Regression metrics
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    cv = cross_val_score(model, self.X, self.y, cv=CROSS_VAL_FOLDS, scoring='r2').mean()
                    logging.info(f"{name} MSE: {mse:.4f}, R2: {r2:.4f}, CV R2: {cv:.4f}")
                    results[name] = {
                        "mse": mse,
                        "r2": r2,
                        "cv_score": cv
                    }

            return results

        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise CustomException(e, sys) from e
