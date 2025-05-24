# Retail Sales Analysis and Forecasting Dashboard

**[Optional: Add a one-sentence tagline for your project here]**

## Table of Contents
1. [Introduction](#introduction)
2. [Project Goal](#project-goal)
3. [Features](#features)
4. [Tech Stack & Dependencies](#tech-stack--dependencies)
5. [Project Structure](#project-structure)
6. [Dataset](#dataset)
7. [Setup and Installation](#setup-and-installation)
8. [Usage Instructions](#usage-instructions)
    - [8.1 Data Preparation](#81-data-preparation)
    - [8.2 Running the Full Analysis Script](#82-running-the-full-analysis-script)
    - [8.3 Running the Interactive Dashboard](#83-running-the-interactive-dashboard)
9. [Methodology](#methodology)
    - [9.1 Data Loading & EDA](#91-data-loading--eda)
    - [9.2 Feature Engineering](#92-feature-engineering)
    - [9.3 Model Training & Evaluation](#93-model-training--evaluation)
    - [9.4 Dashboard Visualization](#94-dashboard-visualization)
10. [Expected Output & Results](#expected-output--results)
11. [Future Enhancements](#future-enhancements)
12. [Contribution](#contribution)
13. [Author & College Information](#author--college-information)

---

## 1. Introduction
This project focuses on analyzing retail sales data to uncover trends, patterns, and insights. It further aims to forecast future sales using various machine learning models. An interactive dashboard is provided to visualize key performance indicators (KPIs), sales trends, and model performance.

This project was developed as part of the `[Your Course Name/Module]` curriculum at `[College Name]`.

## 2. Project Goal
The primary goals of this project are:
- To perform a comprehensive Exploratory Data Analysis (EDA) on retail sales data.
- To engineer relevant features that can help in predicting sales.
- To train and evaluate multiple machine learning regression models for sales forecasting.
- To develop an interactive Streamlit dashboard for visualizing sales data, analytical insights, and machine learning model outputs.
- To provide a practical tool for understanding and predicting retail sales performance.

## 3. Features
- **Data Loading & Preprocessing:** Handles loading of sales data and prepares it for analysis.
- **Exploratory Data Analysis (EDA):** Generates various plots to understand sales distributions, trends over time, and performance by store/category.
- **Feature Engineering:** Creates time-based features, lag features, and rolling window statistics.
- **Machine Learning Model Training:**
    - Trains multiple regression models (e.g., Linear Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM).
    - Evaluates models based on metrics like MAE, RMSE, and R².
- **Interactive Dashboard (Streamlit):**
    - KPI display (Total Sales, Average Sales, etc.).
    - Customizable date range filters.
    - Visualizations of sales trends, monthly performance, store performance, and category distribution.
    - On-demand ML analysis and results display within the dashboard.
    - Data exploration and download capabilities.
- **Sample Data Generation:** If no input `train.csv` is found, a sample dataset can be generated.
- **Dataset Adaptability:** Includes a script to preprocess and adapt external datasets like the "Store Sales - Time Series Forecasting (Favorita)" dataset for use with this project.

## 4. Tech Stack & Dependencies
- **Programming Language:** Python 3.x
- **Core Libraries:**
    - `pandas`: Data manipulation and analysis.
    - `numpy`: Numerical operations.
    - `matplotlib` & `seaborn`: Static data visualization.
    - `scikit-learn`: Machine learning (model training, metrics, preprocessing).
    - `xgboost`: Gradient boosting library.
    - `lightgbm`: Gradient boosting library.
- **Dashboard:**
    - `streamlit`: Web application framework for interactive dashboards.
    - `plotly`: Interactive data visualization.
- **Others:**
    - `datetime`, `os`, `warnings`, `joblib` (for model saving/loading in `utils.py`).

A detailed list of dependencies can be found in `requirements.txt`.

## 5. Project Structure
.
├── train.csv # Main data file (can be generated or user-provided)
├── preprocess_favorita.py # Script to preprocess Favorita dataset (if used)
├── retail_sales_analysis.py # Core script for EDA, feature engineering, and model training
├── dashboard.py # Streamlit dashboard application
├── data_generator.py # Generates sample sales data if train.csv is missing
├── config.py # Configuration settings for the project
├── utils.py # Utility functions (e.g., evaluation, model saving)
├── run_analysis.py # Main script to execute the complete analysis pipeline
├── run_dashboard.py # Script to launch the Streamlit dashboard
├── requirements.txt # Python dependencies
└── README.md # This file


## 6. Dataset
This project is designed to work with a `train.csv` file containing retail sales data. The expected columns are:
- `Date`: Date of the transaction (YYYY-MM-DD, MM/DD/YYYY, or DD/MM/YYYY).
- `Store`: Identifier for the store.
- `Item`: Identifier for the item sold.
- `Category`: Category of the item (e.g., Electronics, Clothing).
- `Promotion`: Binary flag (0 or 1) indicating if a promotion was active.
- `Sales`: The sales amount.

If `train.csv` is not provided, `data_generator.py` will create a synthetic dataset.

This project has also been tested with the **"Store Sales - Time Series Forecasting (Favorita)" dataset** from Kaggle. A preprocessing script (`preprocess_favorita.py`) is included to adapt its `train.csv` file for use with this project structure.
- **Favorita Dataset Source:** [Kaggle - Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- To use it, download the `train.csv` from the Kaggle competition, place it in the project's root directory (or update the path in `preprocess_favorita.py`), and run `python preprocess_favorita.py`.

## 7. Setup and Installation
1.  **Clone the Repository (if applicable) or Download Files:**
    ```bash
    git clone [your-repository-link]
    cd [repository-name]
    ```
    Or ensure all project files are in a single directory.

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    Make sure you have Python 3.x installed. Then, install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## 8. Usage Instructions

### 8.1 Data Preparation
-   **Option 1: Use your own `train.csv`:**
    Place your data file named `train.csv` in the root project directory. Ensure it has the columns mentioned in the [Dataset](#dataset) section.
-   **Option 2: Use the generated sample data:**
    If no `train.csv` is present, the scripts (`run_analysis.py` or `dashboard.py`) will automatically generate a sample `train.csv` using `data_generator.py`.
-   **Option 3: Use the Favorita Dataset:**
    1.  Download `train.csv` from the [Favorita Kaggle competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data).
    2.  Place it in the root directory of this project.
    3.  Run the preprocessing script:
        ```bash
        python preprocess_favorita.py
        ```
        This will create a new `train.csv` in the project root, formatted for this project. **Note:** Consider using the sampling options within `preprocess_favorita.py` for initial runs as the full dataset is very large.

### 8.2 Running the Full Analysis Script
This script performs EDA, feature engineering, trains models, and evaluates them. Outputs (like plots) will be displayed, and insights printed to the console.
```bash
python run_analysis.py 
```
The Config.create_directories() function will create an output/ directory with subdirectories for plots/ and models/, though this template currently doesn't save plots/models by default from run_analysis.py (it shows them). This can be extended.

### 8.3 Running the Interactive Dashboard
This launches a web-based Streamlit dashboard.
```bash
python run_dashboard.py
```

Alternatively, you can run:
```bash
streamlit run dashboard.py
```
Open your web browser and navigate to the local URL provided.

## 9 Methodology
### 9.1 Data Loading & EDA
The data is loaded from train.csv.
Exploratory Data Analysis involves visualizing sales distributions, sales over time, performance by store, monthly sales patterns, and sales by category/item to understand underlying trends and seasonality.

### 9.2 Feature Engineering 
Time-based Features: Year, Month, Day, DayOfWeek, DayOfYear, WeekOfYear, Quarter, IsWeekend.
Lag Features: Sales from previous periods (e.g., 1 day ago, 7 days ago, 30 days ago) for the same store/item combination.
Rolling Window Features: Rolling mean and standard deviation of sales over defined windows (e.g., 7 days, 30 days) for the same store/item.
Categorical Encoding: Label encoding for categorical features like 'Category'.

### 9.3 Model Training & Evaluation
The dataset is split into training and testing sets chronologically.
Features are scaled for models like Linear Regression.
Multiple regression models are trained:
Linear Regression
Random Forest Regressor
Gradient Boosting Regressor
XGBoost Regressor
LightGBM Regressor
Models are evaluated using:
Mean Absolute Error (MAE)
Mean Squared Error (MSE)
Root Mean Squared Error (RMSE)
R-squared (R²)
The best model is identified based on the R² score.

### 9.4 Dashboard Visualization
The Streamlit dashboard provides an interactive interface.
Key metrics are displayed using st.metric.
Plotly is used for interactive charts showing sales trends, distributions, and model predictions.
Users can upload their own data (CSV format), filter by date range, and trigger ML analysis.

## 10. Expected Output & Results
### Console Output (run_analysis.py):
Data loading information, summary statistics, missing value counts.
Key insights from EDA.
Feature engineering steps.
Model training progress.
Model evaluation metrics (MAE, RMSE, R²) for each model.
Identification of the best performing model.
### Plots (run_analysis.py):
EDA plots (sales distribution, sales over time, etc.).
Model comparison plots (R² and RMSE).
Actual vs. Predicted sales plots, residuals plot for the best model.
### Dashboard (dashboard.py):
Interactive display of KPIs.
Dynamic charts based on user selections (date range).
Results of ML analysis performed within the dashboard.
Option to download filtered data and a summary report.

## 11. Future Enhancements
**Hyperparameter Tuning**: Implement GridSearchCV or RandomizedSearchCV for more rigorous hyperparameter optimization of ML models.
**Advanced Feature Engineering**: Incorporate external factors like holidays (using holidays_events.csv from Favorita), economic indicators (like oil.csv), or store-specific details (stores.csv).
Deep Learning Models: Explore LSTMs or other sequence models for time series forecasting.
**Model Deployment**: Package the model and dashboard for easier deployment (e.g., using Docker, cloud platforms).
**Saving Plots & Models**: Enhance run_analysis.py and config.py to systematically save generated plots and trained models to the output/ directory.
**More Sophisticated Error Analysis**: Deeper dive into model residuals and error patterns.
**Cross-Validation**: Implement time-series cross-validation (e.g., TimeSeriesSplit from scikit-learn).
**Confidence Intervals**: Add prediction intervals to the forecasts in the dashboard.

## 12. Contribution
This project is primarily for academic purposes. However, suggestions or contributions are welcome via [mention how: e.g., GitHub Issues/Pull Requests if applicable, or email].

## 13. Author & College Information
Author: [Your Name]
Student ID: [Your Student ID]
College: [Your College Name]
Course/Program: [Your Course/Program Name]
Project Supervisor (if any): [Supervisor's Name/Title]
Date: May 2025 (or the relevant submission date)
