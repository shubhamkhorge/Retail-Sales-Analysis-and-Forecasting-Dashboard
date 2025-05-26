# Configuration file for the retail sales analysis project

import os


class Config:
    """Configuration settings for the retail sales analysis project"""

    # Data settings
    DATA_FILE = "train.csv"
    OUTPUT_DIR = "output"
    PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")
    MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

    # Model settings
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    CV_FOLDS = 5

    # Feature engineering settings
    LAG_FEATURES = [1, 7, 30]  # Days to lag
    ROLLING_WINDOWS = [7, 30]  # Rolling window sizes

    # Model hyperparameters
    MODEL_PARAMS = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [1, 2],
        },
        "XGBoost": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 1.0],
        },
        "LightGBM": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 6, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "num_leaves": [31, 50, 100],
        },
    }

    # Visualization settings
    FIGURE_SIZE = (12, 8)
    DPI = 300
    STYLE = "seaborn-v0_8"

    # Evaluation metrics
    METRICS = ["MAE", "MSE", "RMSE", "R2", "MAPE"]

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.PLOTS_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        print("✓ Directories created")

    @classmethod
    def get_model_params(cls, model_name):
        """Get hyperparameters for a specific model"""
        return cls.MODEL_PARAMS.get(model_name, {})
