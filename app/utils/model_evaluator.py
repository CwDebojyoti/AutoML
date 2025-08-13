# model_evaluator.py
import os
import json
import sys
import logging
import datetime
import numpy as np
import pandas as pd
import joblib
from io import BytesIO
from google.cloud import storage
import tempfile
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
    mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score

from app.config import REPORT_DIR, RANDOM_STATE, CROSS_VAL_FOLDS
from app.exception_logging.exception import CustomException

os.makedirs(REPORT_DIR, exist_ok=True)


class ModelEvaluator:
    def __init__(self, is_classification: bool):
        self.is_classification = is_classification

    def _classification_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "classification_report": classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        }
        if y_proba is not None:
            try:
                classes = np.unique(y_true)
                if len(classes) > 2:
                    y_true_bin = label_binarize(y_true, classes=classes)
                    metrics["roc_auc_ovr"] = roc_auc_score(y_true_bin, y_proba, average="macro", multi_class="ovr")
                else:
                    p = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    metrics["roc_auc"] = roc_auc_score(y_true, p)
                    metrics["average_precision"] = average_precision_score(y_true, p)
            except Exception as e:
                logging.warning(f"ROC/AUC computation failed: {e}")
        return metrics

    def _regression_metrics(self, y_true, y_pred):
        return {
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "explained_variance": explained_variance_score(y_true, y_pred)
        }

    def _plot_confusion_matrix(self, y_true, y_pred, labels, save_path):
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap=plt.cm.Blues)
        ax.figure.colorbar(im)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        plt.tight_layout()
        fig.savefig(save_path)
        plt.close(fig)
        return save_path

    def evaluate_and_save(self, model_name, model, X_train, y_train, X_test, y_test):
        try:
            
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            """
            save_dir = os.path.join(REPORT_DIR, f"{model_name}_{ts}")
            os.makedirs(save_dir, exist_ok=True)
            """

            y_pred = model.predict(X_test)
            y_proba = None
            if self.is_classification and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                except:
                    pass

            # Metrics
            if self.is_classification:
                metrics = self._classification_metrics(y_test, y_pred, y_proba)
                labels = np.unique(np.concatenate([y_test, y_pred]))
                # cm_path = self._plot_confusion_matrix(y_test, y_pred, labels, os.path.join(save_dir, "confusion.png"))
                cm_path = None
            else:
                metrics = self._regression_metrics(y_test, y_pred)
                cm_path = None

            # CV score
            cv_scores = cross_val_score(model, X_train, y_train, cv=CROSS_VAL_FOLDS,
                                        scoring="accuracy" if self.is_classification else "r2")
            metrics["cv_mean"] = np.mean(cv_scores)

            # Save summary
            summary = {
                "model_name": model_name,
                "timestamp": ts,
                "metrics": metrics,
                "artifacts": {"confusion_matrix": cm_path} if cm_path else {}
            }
            """
            summary_path = os.path.join(save_dir, "summary.json")
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            """
            return summary
        except Exception as e:
            raise CustomException(e, sys)
        
    
    def load_model_from_gcs(self, bucket_name, blob_name):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
        return joblib.load(BytesIO(data))
        
        
    def get_latest_model_blob_name(self, bucket_name, dataset_name, model_name):
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        prefix = f"models/{dataset_name}/"
        blobs = list(bucket.list_blobs(prefix=prefix))
        candidates = [b.name for b in blobs if b.name.startswith(f"{prefix}{model_name}_") and b.name.endswith("_best.pkl")]
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0]

