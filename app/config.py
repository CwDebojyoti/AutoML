import os
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBClassifier, XGBRegressor

# ----------------------------
# File Paths
# ----------------------------
DATA_DIR = "data/"
REPORT_DIR = "reports/"
MODEL_DIR = "models/"
LOG_FILE = "logs/app.log"
GCS_BUCKET_NAME = "deb_automl"

# Ensure required directories exist
for directory in [DATA_DIR, REPORT_DIR, MODEL_DIR, os.path.dirname(LOG_FILE)]:
    os.makedirs(directory, exist_ok=True)

# ----------------------------
# Preprocessing Settings
# ----------------------------
MISSING_VALUE_THRESHOLD = 0.5
CORRELATION_THRESHOLD = 0.95
TARGET_GUESS_LIMIT = 10  # Threshold for classification vs regression decision

# ----------------------------
# Model Training Settings
# ----------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42
CROSS_VAL_FOLDS = 5

# ----------------------------
# Available Models
# ----------------------------
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "SVC": SVC(probability=True),
    "KNeighborsClassifier": KNeighborsClassifier(),
    "XGBClassifier": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

REGRESSORS = {
    "LinearRegression": LinearRegression(),
    "DecisionTreeRegressor": DecisionTreeRegressor(),
    "RandomForestRegressor": RandomForestRegressor(),
    "SVR": SVR(),
    "KNeighborsRegressor": KNeighborsRegressor(),
    "XGBRegressor": XGBRegressor()
}

# ----------------------------
# Grid search parameter grids (used by GridSearchCV)
# ----------------------------
GRID_SEARCH_PARAMS = {
    # --- Classifiers ---
    "LogisticRegression": {
        "C": [0.01, 0.1, 1.0, 10.0],
        "max_iter": [100, 200]
    },
    "DecisionTreeClassifier": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "RandomForestClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 5]
    },
    "SVC": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"]
    },
    "KNeighborsClassifier": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2]  # 1 = Manhattan, 2 = Euclidean
    },
    "XGBClassifier": {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    },

    # --- Regressors ---
    "LinearRegression": {
        "fit_intercept": [True, False]
    },
    "DecisionTreeRegressor": {
        "max_depth": [None, 5, 10, 20],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    },
    "RandomForestRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 5]
    },
    "SVR": {
        "C": [0.1, 1.0, 10.0],
        "kernel": ["rbf", "linear"],
        "gamma": ["scale", "auto"],
        "epsilon": [0.01, 0.1, 0.2]
    },
    "KNeighborsRegressor": {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    },
    "XGBRegressor": {
        "n_estimators": [50, 100],
        "max_depth": [3, 6],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0]
    }
}


# ----------------------------
# Model Parameters
# ----------------------------
MODEL_PARAMS = {
    # --- Classifiers ---
    "LogisticRegression": {
        "C": 1.0,
        "max_iter": 100,
        "solver": "lbfgs"
    },
    "DecisionTreeClassifier": {
        "max_depth": 10,
        "criterion": "gini",
        "random_state": RANDOM_STATE
    },
    "RandomForestClassifier": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE
    },
    "SVC": {
        "C": 1.0,
        "kernel": "rbf",
        "probability": True
    },
    "KNeighborsClassifier": {
        "n_neighbors": 5,
        "weights": "uniform"
    },
    "XGBClassifier": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    },

    # --- Regressors ---
    "LinearRegression": {
        "fit_intercept": True
    },
    "DecisionTreeRegressor": {
        "max_depth": 10,
        "criterion": "squared_error",
        "random_state": RANDOM_STATE
    },
    "RandomForestRegressor": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE
    },
    "SVR": {
        "C": 1.0,
        "kernel": "rbf"
    },
    "KNeighborsRegressor": {
        "n_neighbors": 5,
        "weights": "uniform"
    },
    "XGBRegressor": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 5
    }
}


# ----------------------------
# Model Type Mapping
# ----------------------------
MODEL_TYPE_MAP = {
    "LogisticRegression": "classifier",
    "DecisionTreeClassifier": "classifier",
    "RandomForestClassifier": "classifier",
    "SVC": "classifier",
    "KNeighborsClassifier": "classifier",
    "XGBClassifier": "classifier",
    "LinearRegression": "regressor",
    "DecisionTreeRegressor": "regressor",
    "RandomForestRegressor": "regressor",
    "SVR": "regressor",
    "KNeighborsRegressor": "regressor",
    "XGBRegressor": "regressor"
}

# ----------------------------
# Feature Engineering Settings
# ----------------------------
ENABLE_DATETIME_FEATURES = True
ENABLE_TEXT_FEATURES = False

# ----------------------------
# Report Settings
# ----------------------------
GENERATE_HTML_REPORT = True
GENERATE_PDF_REPORT = False

# ----------------------------
# Logging Setup
# ----------------------------
def configure_logging():
    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
