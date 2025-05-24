# Streamlit dashboard for retail sales analysis

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our analysis modules
from retail_sales_analysis import RetailSalesAnalyzer
from data_generator import generate_retail_data
from utils import validate_data_quality, create_sales_summary
import os

# Page configuration
st.set_page_config(
    page_title="Retail Sales Analytics Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(#f0f2f6, #ffffff);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """Load data with caching, with robust date parsing."""
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            if 'Date' not in df.columns:
                st.error(f"Data file '{file_path}' must contain a 'Date' column.")
                st.warning("Attempting to regenerate sample data due to missing 'Date' column.")
                generate_retail_data(file_path, 10000)  # Smaller dataset for dashboard
                df = pd.read_csv(file_path)
                # Generated data's date format is standard (YYYY-MM-DD)
                df['Date'] = pd.to_datetime(df['Date'])
                return df

            # Robust date parsing
            try:
                df['Date'] = pd.to_datetime(df['Date'])
            except (ValueError, TypeError):
                try:
                    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
                except Exception as e_date_parse:
                    st.error(f"Error parsing 'Date' column in '{file_path}': {e_date_parse}. "
                             "Please check format or delete the file to regenerate sample data.")
                    return None # Indicate failure
            return df
        else:
            st.warning("Data file not found. Generating sample data...")
            generate_retail_data(file_path, 10000)  # Smaller dataset for dashboard
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date']) # Generated data's date format is standard
            return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def prepare_dashboard_data(df):
    """Prepare aggregated data for dashboard"""
    # Daily sales
    daily_sales = df.groupby('Date')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
    daily_sales.columns = ['Date', 'Total_Sales', 'Avg_Sales', 'Transactions']
    
    # Monthly sales
    df['YearMonth'] = df['Date'].dt.to_period('M')
    monthly_sales = df.groupby('YearMonth')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
    monthly_sales['YearMonth'] = monthly_sales['YearMonth'].astype(str)
    
    # Store performance
    store_performance = df.groupby('Store')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
    store_performance.columns = ['Store', 'Total_Sales', 'Avg_Sales', 'Transactions']
    
    # Category analysis (if exists)
    category_sales = None
    if 'Category' in df.columns:
        category_sales = df.groupby('Category')['Sales'].agg(['sum', 'mean', 'count']).reset_index()
        category_sales.columns = ['Category', 'Total_Sales', 'Avg_Sales', 'Transactions']
    
    return daily_sales, monthly_sales, store_performance, category_sales

def create_summary_metrics(df):
    """Create summary metrics for the dashboard"""
    total_sales = df['Sales'].sum()
    avg_sales = df['Sales'].mean()
    total_transactions = len(df)
    unique_stores = df['Store'].nunique() if 'Store' in df.columns else 0
    unique_items = df['Item'].nunique() if 'Item' in df.columns else 0
    date_range_days = 0
    if not df['Date'].empty: # Check if Date column is not empty
        min_d, max_d = df['Date'].min(), df['Date'].max()
        if pd.notna(min_d) and pd.notna(max_d): # Ensure dates are not NaT
             date_range_days = (max_d - min_d).days

    return {
        'total_sales': total_sales,
        'avg_sales': avg_sales,
        'total_transactions': total_transactions,
        'unique_stores': unique_stores,
        'unique_items': unique_items,
        'date_range': date_range_days
    }

def plot_sales_trend(daily_sales):
    """Create sales trend plot"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_sales['Date'],
        y=daily_sales['Total_Sales'],
        mode='lines',
        name='Daily Sales',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title='Daily Sales Trend',
        xaxis_title='Date',
        yaxis_title='Total Sales (₹)',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def plot_monthly_sales(monthly_sales):
    """Create monthly sales bar chart"""
    fig = px.bar(
        monthly_sales,
        x='YearMonth',
        y='sum',
        title='Monthly Sales Performance',
        labels={'sum': 'Total Sales (₹)', 'YearMonth': 'Month'},
        color='sum',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(template='plotly_white')
    return fig

def plot_store_performance(store_performance, top_n=10):
    """Create store performance chart"""
    top_stores = store_performance.nlargest(top_n, 'Total_Sales')
    
    fig = px.bar(
        top_stores,
        x='Store',
        y='Total_Sales',
        title=f'Top {top_n} Stores by Sales',
        labels={'Total_Sales': 'Total Sales (₹)', 'Store': 'Store ID'},
        color='Total_Sales',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(template='plotly_white')
    return fig

def plot_category_distribution(category_sales):
    """Create category distribution pie chart"""
    if category_sales is not None and not category_sales.empty:
        fig = px.pie(
            category_sales,
            values='Total_Sales',
            names='Category',
            title='Sales Distribution by Category'
        )
        fig.update_layout(template='plotly_white')
        return fig
    return None

def plot_sales_distribution(df):
    """Create sales distribution histogram"""
    fig = px.histogram(
        df,
        x='Sales',
        nbins=50,
        title='Sales Distribution',
        labels={'Sales': 'Sales Amount (₹)', 'count': 'Frequency'}
    )
    
    fig.update_layout(template='plotly_white')
    return fig

def run_ml_analysis(df):
    """Run ML analysis and return results"""
    try:
        # Save data temporarily for analysis
        temp_file = 'temp_dashboard_data.csv'
        # Ensure Date is in a standard format before saving for RetailSalesAnalyzer
        df_copy = df.copy()
        if pd.api.types.is_datetime64_any_dtype(df_copy['Date']):
             df_copy['Date'] = df_copy['Date'].dt.strftime('%Y-%m-%d')
        df_copy.to_csv(temp_file, index=False)
        
        # Initialize analyzer
        analyzer = RetailSalesAnalyzer(temp_file) # RetailSalesAnalyzer will re-parse Date
        
        # Load and process data
        # Instead of analyzer.load_data(), we set df and then call feature_engineering
        # The RetailSalesAnalyzer expects its internal df to have 'Date' parsed.
        # Let's ensure it is parsed correctly by its own logic or pre-parsed.
        if not analyzer.load_data(): # This will read temp_file and parse 'Date'
            st.error("Failed to load data into analyzer for ML analysis.")
            if os.path.exists(temp_file):
                os.remove(temp_file)
            return None, None, None

        analyzer.feature_engineering()
        
        # Prepare model data
        X_train, X_test, y_train, y_test, feature_cols = analyzer.prepare_model_data()
        
        # Train models (subset for speed)
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, r2_score
        
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
            'Linear Regression': LinearRegression()
        }
        
        results = {}
        best_y_pred = None # To store predictions of the best model
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred_current = model.predict(X_test)
            
            results[name] = {
                'MAE': mean_absolute_error(y_test, y_pred_current),
                'R2': r2_score(y_test, y_pred_current)
            }
            if best_y_pred is None or results[name]['R2'] > results[max(results, key=lambda k: results[k]['R2'])]['R2']:
                best_y_pred = y_pred_current

        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return results, y_test, best_y_pred # Return predictions of the best model
        
    except Exception as e:
        st.error(f"Error in ML analysis: {e}")
        import traceback
        st.error(traceback.format_exc())
        if 'temp_file' in locals() and os.path.exists(temp_file):
            os.remove(temp_file)
        return None, None, None

def main():
    """Main dashboard function"""
    
    # Header
    st.markdown('<h1 class="main-header">🛍️ Retail Sales Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Dashboard Controls")
    
    # File upload option
    uploaded_file = st.sidebar.file_uploader(
        "Upload your sales data (CSV)",
        type=['csv'],
        help="Upload a CSV file with columns: Date, Store, Item, Sales. Dates can be YYYY-MM-DD, MM/DD/YYYY, or DD/MM/YYYY."
    )
    
    # Load data
    df_initial = None
    if uploaded_file is not None:
        df_initial = pd.read_csv(uploaded_file)
        if 'Date' not in df_initial.columns:
            st.error("Uploaded file must contain a 'Date' column.")
            return
        try:
            # Attempt standard parsing first
            df_initial['Date'] = pd.to_datetime(df_initial['Date'])
        except (ValueError, TypeError):
            try:
                # Attempt parsing with dayfirst=True if standard fails
                df_initial['Date'] = pd.to_datetime(df_initial['Date'], dayfirst=True)
            except Exception as e_date_parse:
                st.error(f"Error parsing 'Date' column from uploaded file: {e_date_parse}. "
                         "Please ensure dates are in a recognizable format.")
                return # Stop if date parsing fails
        st.sidebar.success("Data uploaded successfully!")
    else:
        df_initial = load_data('train.csv') # load_data handles its own parsing
    
    if df_initial is None or df_initial.empty:
        st.error("Could not load data. Please check your file or ensure sample data can be generated.")
        return
    
    df = df_initial.copy() # Work with a copy

    # Data validation
    with st.sidebar.expander("Data Quality Check"):
        quality_info = validate_data_quality(df) # Assuming validate_data_quality is robust
        st.write(f"**Shape:** {quality_info['shape']}")
        st.write(f"**Missing Values:** {quality_info['missing_values']}")
        st.write(f"**Memory Usage:** {quality_info['memory_mb']:.2f} MB")
    
    # Date range filter
    st.sidebar.subheader("Date Range Filter")
    # Ensure 'Date' column is datetime before min/max
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        st.error("Date column is not in a recognized datetime format after loading.")
        return
        
    min_date = df['Date'].min().date()
    max_date = df['Date'].max().date()
    
    date_range_selected = st.sidebar.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    
    # Filter data based on date range
    df_filtered = df.copy()
    if len(date_range_selected) == 2:
        start_date, end_date = date_range_selected
        # Convert to datetime.datetime for comparison if they are not already
        start_date = datetime.combine(start_date, datetime.min.time())
        end_date = datetime.combine(end_date, datetime.max.time())
        
        df_filtered = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    if df_filtered.empty:
        st.warning("No data available for the selected date range.")
        return

    # Prepare dashboard data
    daily_sales, monthly_sales, store_performance, category_sales = prepare_dashboard_data(df_filtered)
    summary_metrics = create_summary_metrics(df_filtered)
    
    # Summary metrics
    st.subheader("📊 Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Sales",
            value=f"₹{summary_metrics['total_sales']:,.2f}",
            delta=f"{summary_metrics['total_transactions']:,} transactions"
        )
    
    with col2:
        st.metric(
            label="Average Sale",
            value=f"₹{summary_metrics['avg_sales']:.2f}",
            delta=f"Per transaction"
        )
    
    with col3:
        st.metric(
            label="Active Stores",
            value=f"{summary_metrics['unique_stores']}",
            delta=f"{summary_metrics['unique_items']} unique items"
        )
    
    with col4:
        st.metric(
            label="Date Range",
            value=f"{summary_metrics['date_range']} days",
            delta="Analysis period"
        )
    
    # Charts section
    st.subheader("📈 Sales Analytics")
    
    # Two columns for charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # Sales trend
        if not daily_sales.empty:
            trend_fig = plot_sales_trend(daily_sales)
            st.plotly_chart(trend_fig, use_container_width=True)
        
        # Store performance
        if not store_performance.empty:
            store_fig = plot_store_performance(store_performance)
            st.plotly_chart(store_fig, use_container_width=True)
    
    with chart_col2:
        # Monthly sales
        if not monthly_sales.empty:
            monthly_fig = plot_monthly_sales(monthly_sales)
            st.plotly_chart(monthly_fig, use_container_width=True)
        
        # Sales distribution
        if not df_filtered.empty:
            dist_fig = plot_sales_distribution(df_filtered)
            st.plotly_chart(dist_fig, use_container_width=True)
    
    # Category analysis (if available)
    if category_sales is not None and not category_sales.empty:
        st.subheader("🏷️ Category Analysis")
        cat_fig = plot_category_distribution(category_sales)
        if cat_fig:
            st.plotly_chart(cat_fig, use_container_width=True)
    
    # Machine Learning Section
    st.subheader("🤖 Machine Learning Insights")
    
    if st.button("Run ML Analysis", type="primary"):
        if df_filtered.empty:
            st.warning("Cannot run ML analysis on empty data (check date filters).")
        else:
            with st.spinner("Training models... This may take a moment..."):
                ml_results, y_test_ml, y_pred_ml = run_ml_analysis(df_filtered.copy()) # Pass a copy
                
                if ml_results:
                    # Display results
                    res_col1, res_col2 = st.columns(2)
                    
                    with res_col1:
                        st.subheader("Model Performance")
                        results_df = pd.DataFrame(ml_results).T.sort_values(by='R2', ascending=False)
                        st.dataframe(results_df, use_container_width=True)
                    
                    with res_col2:
                        st.subheader("Best Model")
                        if not results_df.empty:
                            best_model_name = results_df.index[0]
                            st.success(f"**{best_model_name}**")
                            st.write(f"R² Score: {ml_results[best_model_name]['R2']:.4f}")
                            st.write(f"MAE: ₹{ml_results[best_model_name]['MAE']:.2f}")
                        else:
                            st.warning("No model results to display.")
                    
                    # Prediction vs Actual plot
                    if y_test_ml is not None and y_pred_ml is not None and not y_test_ml.empty:
                        fig_pred = go.Figure()
                        
                        sample_size = min(500, len(y_test_ml))
                        # Ensure y_test_ml is a Series for .iloc and .sample
                        if isinstance(y_test_ml, np.ndarray):
                            y_test_ml_series = pd.Series(y_test_ml)
                            y_pred_ml_series = pd.Series(y_pred_ml)
                        else: # It should be a Series from prepare_model_data
                            y_test_ml_series = y_test_ml
                            y_pred_ml_series = y_pred_ml

                        sample_indices = y_test_ml_series.sample(n=sample_size, random_state=42).index

                        fig_pred.add_trace(go.Scatter(
                            x=y_test_ml_series[sample_indices],
                            y=y_pred_ml_series[sample_indices],
                            mode='markers',
                            name='Predictions',
                            opacity=0.6
                        ))
                        
                        min_val = min(y_test_ml_series.min(), y_pred_ml_series.min())
                        max_val = max(y_test_ml_series.max(), y_pred_ml_series.max())
                        fig_pred.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(color='red', dash='dash')
                        ))
                        
                        fig_pred.update_layout(
                            title='Actual vs Predicted Sales (Sampled)',
                            xaxis_title='Actual Sales (₹)',
                            yaxis_title='Predicted Sales (₹)',
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                else:
                    st.error("ML Analysis did not produce results.")

    # Data table section
    st.subheader("📋 Data Explorer")
    
    with st.expander("View Raw Data Sample"):
        st.dataframe(df_filtered.head(100), use_container_width=True)
    
    # Download section
    st.subheader("📥 Download Data")
    
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        csv = df_filtered.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data",
            data=csv,
            file_name=f"filtered_sales_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with dl_col2:
        store_performance_head_str = "No store data available."
        if store_performance is not None and not store_performance.empty:
             store_performance_head_str = store_performance.nlargest(5, 'Total_Sales').to_string(index=False)

        summary_report = f"""
        SALES SUMMARY REPORT
        Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        Date Range Analyzed: {df_filtered['Date'].min().date()} to {df_filtered['Date'].max().date()}
        
        OVERVIEW:
        - Total Sales: ₹{summary_metrics['total_sales']:,.2f}
        - Average Sale: ₹{summary_metrics['avg_sales']:.2f}
        - Total Transactions: {summary_metrics['total_transactions']:,}
        - Active Stores: {summary_metrics['unique_stores']}
        - Unique Items: {summary_metrics['unique_items']}
        - Analysis Period: {summary_metrics['date_range']} days
        
        TOP PERFORMING STORES (UP TO 5):
        {store_performance_head_str}
        """
        
        st.download_button(
            label="Download Summary Report",
            data=summary_report,
            file_name=f"sales_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain"
        )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**Retail Sales Analytics Dashboard** | "
        "Built with Streamlit & Plotly | "
        f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

if __name__ == "__main__":
    main()