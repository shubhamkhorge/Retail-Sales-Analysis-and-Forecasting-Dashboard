# run_analysis.py (modified part)
import os
from retail_sales_analysis import RetailSalesAnalyzer
from data_generator import generate_retail_data
from config import Config
import pandas as pd # Import pandas for a quick check

def main():
    """Main function to run the retail sales analysis"""

    print("  RETAIL SALES ANALYSIS & FORECASTING")
    print("=" * 50)

    # Create necessary directories
    Config.create_directories()
    print(f"[DEBUG run_analysis.py] Current working directory: {os.getcwd()}")
    print(f"[DEBUG run_analysis.py] Checking for data file: {Config.DATA_FILE}")

    # Check if data file exists, if not generate sample data
    if not os.path.exists(Config.DATA_FILE):
        print(f"[DEBUG run_analysis.py]  Data file '{Config.DATA_FILE}' NOT FOUND.")
        print(" Generating sample retail sales data using data_generator.py...")
        generate_retail_data(Config.DATA_FILE, 50000) # This will create Config.DATA_FILE
        print(f"[DEBUG run_analysis.py]  Sample data generated and saved to '{Config.DATA_FILE}'.")
    else:
        print(f"[DEBUG run_analysis.py]  Using existing data file: {Config.DATA_FILE}.")
        try:
            # Quick check of the existing file
            df_check = pd.read_csv(Config.DATA_FILE, nrows=5)
            print(f"[DEBUG run_analysis.py] First 5 rows of existing '{Config.DATA_FILE}':")
            print(df_check)
            file_size = os.path.getsize(Config.DATA_FILE)
            print(f"[DEBUG run_analysis.py] Size of existing '{Config.DATA_FILE}': {file_size / (1024*1024):.2f} MB")
            if df_check.empty and file_size < 100: # Arbitrary small size check
                 print(f"[DEBUG run_analysis.py] WARNING: Existing '{Config.DATA_FILE}' seems empty or very small. Is this intended?")
        except Exception as e:
            print(f"[DEBUG run_analysis.py] Error trying to peek into existing '{Config.DATA_FILE}': {e}")


    # Initialize and run analysis
    try:
        print(f"[DEBUG run_analysis.py] Initializing RetailSalesAnalyzer with data_path: '{Config.DATA_FILE}'")
        analyzer = RetailSalesAnalyzer(Config.DATA_FILE) # This passes 'train.csv'
        analyzer.run_complete_analysis()

        print("\n Analysis completed successfully!")
        print(f" Check the '{Config.OUTPUT_DIR}' directory for saved outputs")

    except Exception as e:
        print(f" Error during analysis: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback
        print("💡 Try checking your data file or dependencies")

if __name__ == "__main__":
    main()