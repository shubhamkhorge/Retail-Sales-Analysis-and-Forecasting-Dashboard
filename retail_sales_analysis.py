# Main analysis and forecasting script
from config import Config  # To access PLOTS_DIR and MODELS_DIR
import os
from utils import save_model # To use your existing save_model utility

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings("ignore")

# logging
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Machine Learning imports
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

# Set style for plots
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


class RetailSalesAnalyzer:
    def __init__(self, data_path="train.csv"):
        """Initialize the retail sales analyzer"""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()

    def load_data(self):
        """Load and display basic information about the dataset"""
        logging.info("=" * 50)
        logging.info("LOADING DATA")
        logging.info("=" * 50)

        try:
            self.df = pd.read_csv(self.data_path)
            logging.info(f"Dataset loaded successfully from {self.data_path}!")

            if "Date" not in self.df.columns:
                logging.info(
                    f"Error: 'Date' column not found in {self.data_path}. Cannot proceed with analysis."
                )
                return False

            # Robust date parsing: Convert 'Date' column if it's not already datetime
            if not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
                try:
                    self.df["Date"] = pd.to_datetime(self.df["Date"])
                except (ValueError, TypeError):
                    try:
                        self.df["Date"] = pd.to_datetime(self.df["Date"], dayfirst=True)
                    except Exception as e_date:
                        logging.info(
                            f"Error: Could not parse 'Date' column in {self.data_path}. "
                            f"Please check format. Error: {e_date}"
                        )
                        return False  # Stop if 'Date' cannot be parsed

            logging.info(f"Shape: {self.df.shape}")
            logging.info("\nFirst 5 rows:")
            logging.info(self.df.head())
            logging.info("\nDataset Info:")
            self.df.info(verbose=True, show_counts=True)  # More detailed info
            logging.info("\nMissing Values:")
            logging.info(self.df.isnull().sum())
            logging.info("\nSummary Statistics:")
            # Describe only numeric columns, include datetime for date range
            logging.info(self.df.describe(include=[np.number]))
            if "Date" in self.df.columns and pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
                logging.info(f"\nDate Range: {self.df['Date'].min()} to {self.df['Date'].max()}")
            return True
        except FileNotFoundError:
            logging.info(f"Error: {self.data_path} not found!")
            # Create sample data for demonstration
            logging.info("Attempting to create sample data...")
            if self.create_sample_data():  # create_sample_data now returns bool
                logging.info("Sample data created successfully!")
                # logging.info info for sample data
                logging.info(f"Shape: {self.df.shape}")
                logging.info(self.df.head())
                return True
            else:
                logging.info("Failed to create sample data.")
                return False
        except Exception as e:
            logging.info(f"Error loading data: {e}")
            return False

    def create_sample_data(self):
        """Create sample retail sales data for demonstration. Returns True on success."""
        logging.info("Creating sample retail sales data...")
        try:
            np.random.seed(42)
            n_records = 50000

            start_date = datetime(2020, 1, 1)
            end_date = datetime(2023, 12, 31)
            # Ensure dates are pd.Timestamp for consistency
            dates = pd.to_datetime(pd.date_range(start_date, end_date, periods=n_records))

            data = {
                "Date": dates,  # Already datetime
                "Store": np.random.randint(1, 46, n_records),
                "Item": np.random.randint(1, 51, n_records),
                "Sales": np.random.lognormal(mean=3, sigma=1, size=n_records).round(2),
            }

            df_temp = pd.DataFrame(data)
            df_temp["Month"] = df_temp["Date"].dt.month
            df_temp["DayOfWeek"] = df_temp["Date"].dt.dayofweek

            seasonal_multiplier = 1 + 0.3 * np.sin(2 * np.pi * df_temp["Month"] / 12)
            weekend_multiplier = np.where(df_temp["DayOfWeek"].isin([5, 6]), 1.2, 1.0)

            df_temp["Sales"] = df_temp["Sales"] * seasonal_multiplier * weekend_multiplier
            df_temp["Sales"] = df_temp["Sales"].round(2)

            df_temp["Category"] = np.random.choice(["Electronics", "Clothing", "Food", "Home"], n_records)
            df_temp["Promotion"] = np.random.choice([0, 1], n_records, p=[0.8, 0.2])

            self.df = df_temp[["Date", "Store", "Item", "Category", "Promotion", "Sales"]]
            # Ensure 'Date' is datetime, should be already from pd.date_range
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            return True
        except Exception as e:
            logging.info(f"Error creating sample data: {e}")
            self.df = None  # Ensure df is None if creation fails
            return False

    def exploratory_data_analysis(self):
        """Perform comprehensive EDA"""
        if self.df is None or self.df.empty:
            logging.info("No data loaded for EDA.")
            return

        logging.info("\n" + "=" * 50)
        logging.info("EXPLORATORY DATA ANALYSIS")
        logging.info("=" * 50)

        # Ensure Date column is datetime (should be from load_data)
        if "Date" not in self.df.columns or not pd.api.types.is_datetime64_any_dtype(self.df["Date"]):
            logging.info("Date column is missing or not in datetime format for EDA.")
            return

        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Retail Sales Analysis - EDA", fontsize=16)

        # 1. Sales distribution
        if "Sales" in self.df.columns:
            axes[0, 0].hist(self.df["Sales"], bins=50, alpha=0.7, color="skyblue")
            axes[0, 0].set_title("Sales Distribution")
            axes[0, 0].set_xlabel("Sales")
            axes[0, 0].set_ylabel("Frequency")

        # 2. Sales over time
        daily_sales = self.df.groupby(self.df["Date"].dt.date)["Sales"].sum().reset_index()
        daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])  # Ensure Date is datetime for plotting
        axes[0, 1].plot(daily_sales["Date"], daily_sales["Sales"], alpha=0.7)
        axes[0, 1].set_title("Sales Over Time")
        axes[0, 1].set_xlabel("Date")
        axes[0, 1].set_ylabel("Total Sales")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # 3. Sales by Store (top 10)
        if "Store" in self.df.columns:
            store_sales = self.df.groupby("Store")["Sales"].sum().sort_values(ascending=False).head(10)
            axes[0, 2].bar(
                store_sales.index.astype(str), store_sales.values, color="lightcoral"
            )  # Use store IDs as strings for labels
            axes[0, 2].set_title("Top 10 Stores by Sales")
            axes[0, 2].set_xlabel("Store ID")
            axes[0, 2].set_ylabel("Total Sales")
            axes[0, 2].tick_params(axis="x", rotation=45)

        # 4. Monthly sales pattern
        self.df["Month_Num"] = self.df[
            "Date"
        ].dt.month  # Use a different name to avoid conflict if 'Month' column exists
        monthly_sales = self.df.groupby("Month_Num")["Sales"].mean()
        axes[1, 0].plot(monthly_sales.index, monthly_sales.values, marker="o", color="green")
        axes[1, 0].set_title("Average Sales by Month")
        axes[1, 0].set_xlabel("Month")
        axes[1, 0].set_ylabel("Average Sales")
        axes[1, 0].set_xticks(range(1, 13))

        # 5. Sales by category (if exists)
        if "Category" in self.df.columns:
            cat_sales = self.df.groupby("Category")["Sales"].sum()
            if not cat_sales.empty:
                axes[1, 1].pie(cat_sales.values, labels=cat_sales.index, autopct="%1.1f%%")
                axes[1, 1].set_title("Sales by Category")
        elif "Item" in self.df.columns:  # Alternative if Category doesn't exist
            item_sales = self.df.groupby("Item")["Sales"].sum().sort_values(ascending=False).head(10)
            axes[1, 1].bar(item_sales.index.astype(str), item_sales.values, color="orange")
            axes[1, 1].set_title("Top 10 Items by Sales")
            axes[1, 1].set_xlabel("Item ID")
            axes[1, 1].set_ylabel("Total Sales")
            axes[1, 1].tick_params(axis="x", rotation=45)

        # 6. Day of week pattern
        self.df["DayOfWeekName"] = self.df["Date"].dt.day_name()  # Use a different name
        dow_order = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]
        dow_sales = self.df.groupby("DayOfWeekName")["Sales"].mean().reindex(dow_order)
        axes[1, 2].bar(dow_sales.index, dow_sales.values, color="purple", alpha=0.7)
        axes[1, 2].set_title("Average Sales by Day of Week")
        axes[1, 2].set_xlabel("Day of Week")
        axes[1, 2].set_ylabel("Average Sales")
        axes[1, 2].tick_params(axis="x", rotation=45)

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to prevent suptitle overlap
        try:
                plot_filename = os.path.join(Config.PLOTS_DIR, "eda_summary_plots.png")
                fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
                print(f"✓ EDA summary plot saved to {plot_filename}")
        except Exception as e:
                print(f"✗ Error saving EDA plot: {e}")
        
        plt.show()

        # logging.info key insights
        logging.info("\nKey Insights from EDA:")
        logging.info(f"• Total Sales: ${self.df['Sales'].sum():,.2f}")
        logging.info(
            f"• Average Daily Sales (overall mean): ${self.df['Sales'].mean():.2f}"
        )  # Clarify this is overall transaction mean
        logging.info(f"• Sales Range: ${self.df['Sales'].min():.2f} - ${self.df['Sales'].max():.2f}")
        logging.info(f"• Date Range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        logging.info(f"• Total Days in Data: {(self.df['Date'].max() - self.df['Date'].min()).days}")

    def feature_engineering(self):
        """Create features for machine learning models"""
        if self.df is None or self.df.empty:
            logging.info("No data loaded for feature engineering.")
            return None  # Return None or raise error

        logging.info("\n" + "=" * 50)
        logging.info("FEATURE ENGINEERING")
        logging.info("=" * 50)

        self.processed_df = self.df.copy()

        # Ensure Date column exists and is datetime
        if "Date" not in self.processed_df.columns or not pd.api.types.is_datetime64_any_dtype(
            self.processed_df["Date"]
        ):
            logging.info("Date column is missing or not in datetime format for feature engineering.")
            return None

        # Time-based features
        self.processed_df["Year"] = self.processed_df["Date"].dt.year
        self.processed_df["Month"] = self.processed_df["Date"].dt.month
        self.processed_df["Day"] = self.processed_df["Date"].dt.day
        self.processed_df["DayOfWeek"] = self.processed_df["Date"].dt.dayofweek  # Numeric (0=Mon, 6=Sun)
        self.processed_df["DayOfYear"] = self.processed_df["Date"].dt.dayofyear
        self.processed_df["WeekOfYear"] = self.processed_df["Date"].dt.isocalendar().week.astype(int)
        self.processed_df["Quarter"] = self.processed_df["Date"].dt.quarter
        self.processed_df["IsWeekend"] = (self.processed_df["DayOfWeek"] >= 5).astype(int)

        logging.info("✓ Time-based features created")

        # Sort by date and store/item for lag features
        # Ensure Store and Item exist before trying to group by them
        group_cols = []
        if "Store" in self.processed_df.columns:
            group_cols.append("Store")
        if "Item" in self.processed_df.columns:
            group_cols.append("Item")

        if group_cols and "Sales" in self.processed_df.columns:  # Check if 'Sales' exists
            self.processed_df = self.processed_df.sort_values(group_cols + ["Date"])

            for lag in [1, 7, 30]:
                self.processed_df[f"Sales_Lag_{lag}"] = self.processed_df.groupby(group_cols)["Sales"].shift(
                    lag
                )

            for window in [7, 30]:
                # Need to handle potential issues if a group is smaller than the window
                self.processed_df[f"Sales_Rolling_Mean_{window}"] = self.processed_df.groupby(group_cols)[
                    "Sales"
                ].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
                self.processed_df[f"Sales_Rolling_Std_{window}"] = self.processed_df.groupby(group_cols)[
                    "Sales"
                ].transform(lambda x: x.rolling(window=window, min_periods=1).std())
            logging.info("✓ Lag and rolling features created")
        else:
            logging.info(
                "! Skipping lag/rolling features: 'Store', 'Item', or 'Sales' column missing or group_cols empty."
            )

        # Encode categorical variables
        # Exclude already created time features like 'Month', 'DayOfWeekName', etc. if they were object type
        categorical_cols = self.processed_df.select_dtypes(include=["object", "category"]).columns.tolist()
        # Original date column might be object if not parsed, but we assume it's datetime now.
        # Also, ensure we don't re-encode 'Date' if it was string and somehow missed earlier parsing
        # and not to encode helper columns like 'DayOfWeekName', 'Month_Num' if they are objects
        # Let's be specific about which categorical columns to encode if they exist:
        cols_to_encode = ["Category"]  # Add other known categorical features like 'Store_Type' if any

        for col in cols_to_encode:
            if col in self.processed_df.columns:
                le = LabelEncoder()
                self.processed_df[f"{col}_Encoded"] = le.fit_transform(self.processed_df[col].astype(str))
                logging.info(f"✓ Encoded {col}")

        # Fill missing values created by lag/rolling features (use 0 or mean/median)
        # Backward fill first, then forward fill, then fill with 0 for any remaining NaNs
        self.processed_df = self.processed_df.fillna(method="bfill").fillna(method="ffill").fillna(0)

        logging.info(f"\nFinal processed dataset shape: {self.processed_df.shape}")
        logging.info("Feature engineering completed!")

        return self.processed_df

    def prepare_model_data(self):
        """Prepare data for machine learning models"""
        if self.processed_df is None or self.processed_df.empty:
            logging.info("No processed data available for model preparation.")
            return None, None, None, None, None  # Return Nones

        logging.info("\n" + "=" * 50)
        logging.info("PREPARING MODEL DATA")
        logging.info("=" * 50)

        # Ensure 'Sales' and 'Date' columns exist
        if "Sales" not in self.processed_df.columns:
            logging.info("Error: 'Sales' (target variable) not found in processed data.")
            return None, None, None, None, None
        if "Date" not in self.processed_df.columns or not pd.api.types.is_datetime64_any_dtype(
            self.processed_df["Date"]
        ):
            logging.info("Error: 'Date' column for splitting not found or not datetime in processed data.")
            return None, None, None, None, None

        # Select features for modeling
        # Exclude original Date, original Sales, and any non-numeric/non-encoded categoricals
        # Also exclude temporary columns like Month_Num, DayOfWeekName if created
        feature_cols = [
            col
            for col in self.processed_df.columns
            if col
            not in [
                "Date",
                "Sales",
                "Month_Num",
                "DayOfWeekName",
                "Category",
            ]  # Exclude original category if encoded
            and (
                self.processed_df[col].dtype in [np.int64, np.float64, np.int32, np.float32]  # Numeric
                or col.endswith("_Encoded")
            )  # Or is an encoded categorical
        ]

        # Ensure no NaN in feature_cols (should be handled by fillna in feature_engineering)
        X = self.processed_df[feature_cols].copy()
        X = X.fillna(0)  # Final safety fill for X

        y = self.processed_df["Sales"].copy()
        y = y.fillna(y.mean())  # Fill target NaNs with mean, or drop rows

        if X.empty or y.empty:
            logging.info("Error: Feature set X or target y is empty after selection.")
            return None, None, None, None, None

        logging.info(f"Selected features ({len(feature_cols)}): {feature_cols}")

        # Time-based split (last 20% for testing)
        # Ensure data is sorted by date for a chronological split
        self.processed_df = self.processed_df.sort_values("Date")
        X = X.loc[self.processed_df.index]  # Reorder X according to sorted processed_df
        y = y.loc[self.processed_df.index]  # Reorder y

        split_index = int(len(self.processed_df) * 0.8)

        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]

        if X_train.empty or X_test.empty:
            logging.info(
                "Error: Training or testing set is empty after split. Check data size and split logic."
            )
            return None, None, None, None, None

        logging.info(f"Training set: {X_train.shape}, {y_train.shape}")
        logging.info(f"Testing set: {X_test.shape}, {y_test.shape}")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_models(self, X_train, X_test, y_train, y_test):
        """Train multiple machine learning models"""
        if X_train is None or X_test is None or y_train is None or y_test is None:
            logging.info("Model training skipped due to missing data.")
            return y_test  # Or None, depending on how you want to handle

        logging.info("\n" + "=" * 50)
        logging.info("TRAINING MODELS")
        logging.info("=" * 50)

        # Scale features for linear models (ensure no NaN before scaling)
        X_train_safe = X_train.fillna(0)
        X_test_safe = X_test.fillna(0)

        X_train_scaled = self.scaler.fit_transform(X_train_safe)
        X_test_scaled = self.scaler.transform(X_test_safe)

        # Initialize models
        models_config = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=20,
                min_samples_leaf=5,
            ),  # Tuned slightly
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, random_state=42, max_depth=5, min_samples_leaf=5
            ),  # Tuned slightly
            "XGBoost": xgb.XGBRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=5,
                learning_rate=0.1,
            ),  # Tuned slightly
            "LightGBM": lgb.LGBMRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=5,
                learning_rate=0.1,
                verbose=-1,
            ),  # Tuned slightly
        }

        self.models = {}  # Reset models
        self.results = {}  # Reset results

        for name, model_instance in models_config.items():
            logging.info(f"\nTraining {name}...")
            try:
                if name == "Linear Regression":
                    model_instance.fit(X_train_scaled, y_train)
                    y_pred = model_instance.predict(X_test_scaled)
                else:
                    model_instance.fit(X_train_safe, y_train)  # Use non-scaled for tree models
                    y_pred = model_instance.predict(X_test_safe)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                self.models[name] = model_instance
                self.results[name] = {
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "R2": r2,
                    "predictions": y_pred,
                }

                logging.info(f"✓ {name} trained successfully. MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

                try:
                    model_filename = f"{name.replace(' ', '_')}_model.joblib" # e.g., Random_Forest_model.joblib
                    # We use the save_model function from utils.py,
                    # which uses Config.MODELS_DIR by default if we modify it, or we pass it.
                    # For now, let's assume utils.save_model is as defined and handles directory creation.
                    # We will pass the directory from Config.
                    save_model(model_instance, model_filename, directory=Config.MODELS_DIR)
                except Exception as e:
                    print(f"✗ Error saving model {name}: {e}")

            except Exception as e:
                logging.info(f"✗ Error training {name}: {e}")

        return y_test  # y_test is returned for consistency, though not modified here

    def evaluate_models(self):
        """Compare model performances"""
        if not self.results:
            logging.info("No model results to evaluate.")
            return None

        logging.info("\n" + "=" * 50)
        logging.info("MODEL EVALUATION")
        logging.info("=" * 50)

        results_df = pd.DataFrame(self.results).T
        # Ensure only numeric metric columns are selected for sorting and display
        metric_cols = ["MAE", "MSE", "RMSE", "R2"]
        display_results_df = results_df[metric_cols].copy().sort_values(by="R2", ascending=False)
        display_results_df = display_results_df.round(4)

        logging.info("Model Performance Comparison:")
        logging.info(display_results_df)

        best_model_name = None
        if not display_results_df.empty:
            best_model_name = display_results_df["R2"].idxmax()
            logging.info(f"\nBest Model (by R²): {best_model_name}")
            logging.info(f"Best R² Score: {display_results_df.loc[best_model_name, 'R2']:.4f}")

        # Visualize model comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(display_results_df)))

        # R² comparison
        r2_scores = display_results_df["R2"]
        bars1 = axes[0].bar(r2_scores.index, r2_scores.values, color=colors)
        axes[0].set_title("Model Comparison - R² Score")
        axes[0].set_ylabel("R² Score")
        axes[0].tick_params(axis="x", rotation=45)
        for bar in bars1:
            yval = bar.get_height()
            axes[0].text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.01,
                f"{yval:.3f}",
                ha="center",
                va="bottom",
            )

        # RMSE comparison
        rmse_scores = display_results_df["RMSE"]
        bars2 = axes[1].bar(rmse_scores.index, rmse_scores.values, color=colors)
        axes[1].set_title("Model Comparison - RMSE")
        axes[1].set_ylabel("RMSE")
        axes[1].tick_params(axis="x", rotation=45)
        for bar in bars2:
            yval = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                yval + 0.5,
                f"{yval:.1f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()

        try:
            plot_filename = os.path.join(Config.PLOTS_DIR, "model_performance_comparison.png")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Model comparison plot saved to {plot_filename}")
        except Exception as e:
            print(f"✗ Error saving model comparison plot: {e}")

        plt.show()
        

        return best_model_name

    def visualize_predictions(self, y_test, best_model_name):
        """Visualize actual vs predicted sales"""
        if best_model_name is None or best_model_name not in self.results or y_test is None:
            logging.info(
                "Cannot visualize predictions: Best model not identified, results missing, or y_test missing."
            )
            return

        logging.info("\n" + "=" * 50)
        logging.info(f"SALES FORECASTING VISUALIZATION ({best_model_name})")
        logging.info("=" * 50)

        best_predictions = self.results[best_model_name]["predictions"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Sales Forecasting Results - {best_model_name}", fontsize=16)

        # 1. Actual vs Predicted scatter plot (sample for performance if large)
        sample_size = min(1000, len(y_test))
        sample_indices = (
            np.random.choice(y_test.index, sample_size, replace=False)
            if len(y_test) > sample_size
            else y_test.index
        )

        axes[0, 0].scatter(
            y_test.loc[sample_indices],
            best_predictions[y_test.index.get_indexer(sample_indices)],
            alpha=0.6,
            color="blue",
            s=10,
        )
        min_val = min(y_test.min(), best_predictions.min())
        max_val = max(y_test.max(), best_predictions.max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], "r--", lw=2)
        axes[0, 0].set_xlabel("Actual Sales")
        axes[0, 0].set_ylabel("Predicted Sales")
        axes[0, 0].set_title("Actual vs Predicted Sales (Sampled)")

        # 2. Residuals plot
        residuals = y_test - best_predictions
        axes[0, 1].scatter(
            best_predictions[y_test.index.get_indexer(sample_indices)],
            residuals.loc[sample_indices],
            alpha=0.6,
            color="green",
            s=10,
        )
        axes[0, 1].axhline(y=0, color="r", linestyle="--")
        axes[0, 1].set_xlabel("Predicted Sales")
        axes[0, 1].set_ylabel("Residuals")
        axes[0, 1].set_title("Residuals Plot (Sampled)")

        # 3. Time series plot (sample for clarity)
        # Assuming y_test index corresponds to time if data was sorted chronologically
        time_plot_indices = (
            y_test.loc[sample_indices].sort_index().index
        )  # Plot sampled points in chronological order

        axes[1, 0].plot(
            range(len(time_plot_indices)),
            y_test.loc[time_plot_indices],
            label="Actual",
            alpha=0.7,
            linewidth=2,
        )
        axes[1, 0].plot(
            range(len(time_plot_indices)),
            best_predictions[y_test.index.get_indexer(time_plot_indices)],
            label="Predicted",
            alpha=0.7,
            linewidth=2,
            linestyle="--",
        )
        axes[1, 0].set_xlabel("Time Index (Sampled, Chronological)")
        axes[1, 0].set_ylabel("Sales")
        axes[1, 0].set_title("Time Series: Actual vs Predicted (Sampled)")
        axes[1, 0].legend()

        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=50, alpha=0.7, color="orange", density=True)
        axes[1, 1].set_xlabel("Prediction Error (Actual - Predicted)")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Error Distribution")
        axes[1, 1].axvline(x=0, color="r", linestyle="--")

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        try:
            plot_filename = os.path.join(Config.PLOTS_DIR, f"{best_model_name.replace(' ', '_')}_prediction_visualization.png")
            fig.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Prediction visualization plot for {best_model_name} saved to {plot_filename}")
        except Exception as e:
            print(f"✗ Error saving prediction visualization plot: {e}")


        plt.show()

        logging.info(f"\nForecasting Summary ({best_model_name}):")
        logging.info(f"• MAE: ${self.results[best_model_name]['MAE']:.2f}")
        logging.info(f"• RMSE: ${self.results[best_model_name]['RMSE']:.2f}")
        logging.info(f"• R²: {self.results[best_model_name]['R2']:.4f}")
        logging.info(f"• Mean Prediction Error: ${residuals.mean():.2f}")  # (Actual - Predicted)
        logging.info(f"• Std Prediction Error: ${residuals.std():.2f}")

    def run_complete_analysis(self):
        """Run the complete retail sales analysis pipeline"""
        logging.info("RETAIL SALES ANALYSIS AND FORECASTING")
        logging.info("=" * 60)

        if not self.load_data():
            logging.info("❌ Halting analysis due to data loading issues.")
            return

        self.exploratory_data_analysis()

        if self.feature_engineering() is None:
            logging.info("❌ Halting analysis due to feature engineering issues.")
            return

        model_data = self.prepare_model_data()
        if any(data is None for data in model_data):  # Check if any returned item is None
            logging.info("❌ Halting analysis due to model data preparation issues.")
            return
        X_train, X_test, y_train, y_test, _ = model_data  # Unpack

        y_test_returned = self.train_models(X_train, X_test, y_train, y_test)
        if not self.results:  # Check if any model trained successfully
            logging.info("❌ Halting analysis as no models were trained successfully.")
            return

        best_model_name = self.evaluate_models()
        if best_model_name:
            self.visualize_predictions(y_test_returned, best_model_name)  # Use y_test from train_models return
        else:
            logging.info("⚠️ Skipping prediction visualization as no best model was determined.")

        logging.info("\n" + "=" * 60)
        logging.info("ANALYSIS COMPLETED!")
        logging.info("=" * 60)


# Main execution
if __name__ == "__main__":
    analyzer = RetailSalesAnalyzer("train.csv")
    analyzer.run_complete_analysis()
