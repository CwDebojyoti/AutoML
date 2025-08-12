import sys
import os
import logging
import numpy as np
import pandas as pd
import joblib
import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from app.exception_logging.exception import CustomException
from app.config import (
    CLASSIFIERS, REGRESSORS, MODEL_PARAMS, GRID_SEARCH_PARAMS,
    TEST_SIZE, RANDOM_STATE, CROSS_VAL_FOLDS, MODEL_DIR
)


class ModelTrainer:
    def __init__(self, X, y, use_grid_search: bool = True, n_jobs: int = -1, verbose: int =1 ):

        """
        X: features (can be numpy array, pandas DataFrame, or sparse matrix)
        y: target (1d)
        use_gridsearch: if True, run GridSearchCV when a grid exists for the model
        n_jobs: parallel jobs for GridSearch and model fit (where supported)
        verbose: verbosity for GridSearch
        """
        self.X = X
        self.y = y
        self.use_grid_search = use_grid_search
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.is_classification = self._determine_task_type()

    def _determine_task_type(self):
        """Decide whether the task is classification or regression based on target values."""
        if pd.api.types.is_numeric_dtype(self.y):
            unique_values = self.y.nunique()
            if unique_values <= 10:
                logging.info(f"Detected classification task (target has {unique_values} unique values).")
                return True
            else:
                logging.info(f"Detected regression task with unique target values = {unique_values}.")
                return False
        else:
            logging.info("Detected classification task (non-numeric target).")
            return True
        
    def _get_cv(self):
        if self.is_classification:
            return StratifiedKFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        return KFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    def _eval_classification(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        return {"accuracy": acc, "report": report}
    
    def _eval_regression(self, model, X_test, y_test):
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        return {"mse": mse, "r2": r2}
    
    def _maybe_set_params(self, model, name):
        """Set custom parameters for the model if available."""
        if name in MODEL_PARAMS and isinstance(MODEL_PARAMS[name], dict) and MODEL_PARAMS[name]:
            try:
                model.set_params(**MODEL_PARAMS[name])
                logging.info(f"Applied custom params for {name}: {MODEL_PARAMS[name]}")
            except Exception as e:
                logging.warning(f"Could not set MODEL_PARAMS for {name}: {e}")
            

    def train_models(self, dataset_name):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=self.y if self.is_classification else None
            )

            models = CLASSIFIERS if self.is_classification else REGRESSORS
            cv = self._get_cv()
            results = {}

            # Get timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            for name, model in models.items():
                logging.info(f"Training model: {name}")
                best_model = None
                best_score = None
                details = {}
                
                # If using grid search, set up the grid
                grid = GRID_SEARCH_PARAMS.get(name, None) if isinstance(GRID_SEARCH_PARAMS, dict) else None

                if self.use_grid_search and grid:
                    logging.info(f"Using GridSearchCV for {name} with params: {grid}")
                    grid_search = GridSearchCV(
                        estimator=model,
                        param_grid=grid,
                        cv=cv,
                        n_jobs=self.n_jobs,
                        verbose=self.verbose,
                        scoring='accuracy' if self.is_classification else 'neg_mean_squared_error'
                    )

                    grid_search.fit(X_train, y_train)
                    best_model = grid_search.best_estimator_
                    best_score = grid_search.best_score_
                    details['grid_search_best_params'] = grid_search.best_params_
                    logging.info(f"GridSearch best_score (CV): {best_score:.4f}")
                else:
                    self._maybe_set_params(model, name)
                    model.fit(X_train, y_train)
                    best_model = model

                if self.is_classification:
                    eval_metrics = self._eval_classification(best_model, X_test, y_test)
                else:
                    eval_metrics = self._eval_regression(best_model, X_test, y_test)

                details.update(eval_metrics)
                results[name] = details

                model_fname = os.path.join(MODEL_DIR, f"{name}_{dataset_name}_{timestamp}_best.pkl")
                try:
                    joblib.dump(best_model, model_fname)
                    logging.info(f"Saved {name} to {model_fname}")
                except Exception as e:
                    logging.warning(f"Failed to save model {name}: {e}") 


                # Apply custom params if available
                """
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
                """

            return results, X_train, X_test, y_train, y_test

        except Exception as e:
            logging.error(f"Model training failed: {str(e)}")
            raise CustomException(e, sys) from e
