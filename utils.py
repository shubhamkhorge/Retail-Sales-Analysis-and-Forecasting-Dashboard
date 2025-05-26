# Utility functions for the retail sales analysis project

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculate comprehensive evaluation metrics"""
    metrics = {
        "Model": model_name,
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAPE": calculate_mape(y_true, y_pred),
    }
    return metrics


def save_model(model, filename, directory="models"):
    """Save trained model to disk"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    joblib.dump(model, filepath)
    print(f"✓ Model saved to {filepath}")


def load_model(filename, directory="models"):
    """Load trained model from disk"""
    filepath = os.path.join(directory, filename)
    if os.path.exists(filepath):
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model
    else:
        print(f"✗ Model file not found: {filepath}")
        return None


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Plot feature importance for tree-based models"""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]

        plt.figure(figsize=(12, 8))
        plt.title(f"Top {top_n} Feature Importances")
        plt.bar(range(top_n), importances[indices])
        plt.xticks(
            range(top_n), [feature_names[i] for i in indices], rotation=45, ha="right"
        )
        plt.ylabel("Importance")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
    else:
        print("Model doesn't have feature_importances_ attribute")


def create_sales_summary(df, date_col="Date", sales_col="Sales"):
    """Create a comprehensive sales summary"""
    summary = {
        "Total Sales": df[sales_col].sum(),
        "Average Daily Sales": df[sales_col].mean(),
        "Median Sales": df[sales_col].median(),
        "Sales Std Dev": df[sales_col].std(),
        "Min Sales": df[sales_col].min(),
        "Max Sales": df[sales_col].max(),
        "Total Records": len(df),
        "Date Range": f"{df[date_col].min().date()} to {df[date_col].max().date()}",
    }
    return summary


def detect_outliers(df, column, method="iqr", threshold=1.5):
    """Detect outliers in a column using IQR or Z-score method"""
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    elif method == "zscore":
        z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
        outliers = df[z_scores > threshold]
    else:
        raise ValueError("Method must be 'iqr' or 'zscore'")

    return outliers


def plot_sales_trends(df, date_col="Date", sales_col="Sales", freq="D", save_path=None):
    """Plot sales trends over time"""
    # Aggregate by frequency
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    if freq == "D":
        trend_data = df_copy.groupby(df_copy[date_col].dt.date)[sales_col].sum()
        title = "Daily Sales Trend"
    elif freq == "W":
        trend_data = df_copy.groupby(df_copy[date_col].dt.to_period("W"))[
            sales_col
        ].sum()
        title = "Weekly Sales Trend"
    elif freq == "M":
        trend_data = df_copy.groupby(df_copy[date_col].dt.to_period("M"))[
            sales_col
        ].sum()
        title = "Monthly Sales Trend"
    else:
        raise ValueError("Frequency must be 'D', 'W', or 'M'")

    plt.figure(figsize=(15, 6))
    plt.plot(trend_data.index, trend_data.values, linewidth=2)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_forecast_report(y_true, y_pred, model_name, save_path=None):
    """Generate a comprehensive forecast report"""
    metrics = evaluate_model(y_true, y_pred, model_name)

    report = f"""
    SALES FORECASTING REPORT
    ========================
    
    Model: {model_name}
    Test Period: {len(y_true)} observations
    
    PERFORMANCE METRICS:
    -------------------
    Mean Absolute Error (MAE): ${metrics['MAE']:,.2f}
    Root Mean Squared Error (RMSE): ${metrics['RMSE']:,.2f}
    R-squared (R²): {metrics['R2']:.4f}
    Mean Absolute Percentage Error (MAPE): {metrics['MAPE']:.2f}%
    
    SALES STATISTICS:
    ----------------
    Actual Sales - Mean: ${y_true.mean():,.2f}
    Actual Sales - Std: ${y_true.std():,.2f}
    Predicted Sales - Mean: ${y_pred.mean():,.2f}
    Predicted Sales - Std: ${y_pred.std():,.2f}
    
    FORECAST ACCURACY:
    -----------------
    Mean Prediction Error: ${(y_pred - y_true).mean():,.2f}
    Prediction Error Std: ${(y_pred - y_true).std():,.2f}
    
    MODEL INTERPRETATION:
    --------------------
    • R² of {metrics['R2']:.3f} means the model explains {metrics['R2']*100:.1f}% of sales variance
    • On average, predictions are off by ${metrics['MAE']:,.2f} (MAE)
    • MAPE of {metrics['MAPE']:.1f}% indicates prediction accuracy
    """

    print(report)

    if save_path:
        with open(save_path, "w") as f:
            f.write(report)
        print(f"✓ Report saved to {save_path}")

    return metrics


def create_prediction_intervals(y_pred, residuals, confidence=0.95):
    """Create prediction intervals for forecasts"""
    # Calculate residual standard deviation
    residual_std = np.std(residuals)

    # Calculate z-score for confidence interval
    from scipy.stats import norm

    z_score = norm.ppf((1 + confidence) / 2)

    # Calculate intervals
    margin_error = z_score * residual_std
    lower_bound = y_pred - margin_error
    upper_bound = y_pred + margin_error

    return lower_bound, upper_bound


def validate_data_quality(df, required_columns=None):
    """Validate data quality and completeness"""
    print("DATA QUALITY REPORT")
    print("=" * 30)

    # Check required columns
    if required_columns:
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            print(f"✗ Missing required columns: {missing_cols}")
        else:
            print("✓ All required columns present")

    # Check data types
    print(f"\nDataset shape: {df.shape}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

    # Check missing values
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print(f"\n✗ Missing values found:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"  {col}: {count} ({count/len(df)*100:.1f}%)")
    else:
        print("\n✓ No missing values")

    # Check duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"\n✗ {duplicates} duplicate rows found")
    else:
        print("\n✓ No duplicate rows")

    # Check date column if exists
    date_cols = df.select_dtypes(include=["datetime64"]).columns
    if len(date_cols) > 0:
        for col in date_cols:
            print(f"\n✓ Date range for {col}: {df[col].min()} to {df[col].max()}")

    return {
        "shape": df.shape,
        "missing_values": missing_data.sum(),
        "duplicates": duplicates,
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
    }
