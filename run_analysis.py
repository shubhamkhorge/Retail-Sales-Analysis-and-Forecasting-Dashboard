# Simple script to run the complete retail sales analysis

from retail_sales_analysis import RetailSalesAnalyzer
from data_generator import generate_retail_data
from config import Config
import os

def main():
    """Main function to run the retail sales analysis"""
    
    print("🛍️  RETAIL SALES ANALYSIS & FORECASTING")
    print("=" * 50)
    
    # Create necessary directories
    Config.create_directories()
    
    # Check if data file exists, if not generate sample data
    if not os.path.exists(Config.DATA_FILE):
        print(f"📁 Data file '{Config.DATA_FILE}' not found.")
        print("🔄 Generating sample retail sales data...")
        generate_retail_data(Config.DATA_FILE, 50000)
        print("✅ Sample data generated successfully!\n")
    else:
        print(f"📁 Using existing data file: {Config.DATA_FILE}\n")
    
    # Initialize and run analysis
    try:
        analyzer = RetailSalesAnalyzer(Config.DATA_FILE)
        analyzer.run_complete_analysis()
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📊 Check the '{Config.OUTPUT_DIR}' directory for saved outputs")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        print("💡 Try checking your data file or dependencies")

if __name__ == "__main__":
    main()