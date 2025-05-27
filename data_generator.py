# Script to generate sample retail sales data if train.csv is not available

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_retail_data(filename="train.csv", n_records=50000):
    """
    Generate sample retail sales data

    Parameters:
    filename (str): Output filename
    n_records (int): Number of records to generate
    """
    print(f"Generating {n_records} sample retail sales records...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Date range (4 years of data)
    start_date = datetime(2020, 1, 1)
    end_date = datetime(2023, 12, 31)
    dates = pd.date_range(start_date, end_date, periods=n_records)

    # Generate base data
    data = {
        "Date": dates,
        "Store": np.random.randint(1, 46, n_records),  # 45 stores
        "Item": np.random.randint(1, 51, n_records),  # 50 items
        "Sales": np.random.lognormal(mean=3, sigma=1, size=n_records),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Add time-based features for seasonality
    df["Month"] = df["Date"].dt.month
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Quarter"] = df["Date"].dt.quarter

    # Add seasonal patterns
    # Holiday boost (December)
    holiday_boost = np.where(df["Month"] == 12, 1.5, 1.0)

    # Summer boost (June-August)
    summer_boost = np.where(df["Month"].isin([6, 7, 8]), 1.2, 1.0)

    # Weekend boost
    weekend_boost = np.where(df["DayOfWeek"].isin([5, 6]), 1.3, 1.0)

    # Apply seasonal patterns
    df["Sales"] = df["Sales"] * holiday_boost * summer_boost * weekend_boost

    # Add categorical features
    categories = ["Electronics", "Clothing", "Food", "Home", "Sports"]
    df["Category"] = np.random.choice(categories, n_records)

    # Add promotion flag (20% of sales have promotions)
    df["Promotion"] = np.random.choice([0, 1], n_records, p=[0.8, 0.2])

    # Promotion boost
    promo_boost = np.where(df["Promotion"] == 1, 1.4, 1.0)
    df["Sales"] = df["Sales"] * promo_boost

    # Round sales to 2 decimal places
    df["Sales"] = df["Sales"].round(2)

    # Add some realistic constraints
    # High-value items (Electronics) should have higher average sales
    electronics_mask = df["Category"] == "Electronics"
    df.loc[electronics_mask, "Sales"] *= 1.5

    # Food items should have lower variance
    food_mask = df["Category"] == "Food"
    df.loc[food_mask, "Sales"] = df.loc[food_mask, "Sales"] * 0.7 + 20

    # Final adjustments
    df["Sales"] = df["Sales"].round(2)
    df = df[df["Sales"] > 0]  # Remove any negative sales

    # Select final columns
    final_df = df[["Date", "Store", "Item", "Category", "Promotion", "Sales"]]

    # Sort by date
    final_df = final_df.sort_values("Date").reset_index(drop=True)

    # Save to CSV
    final_df.to_csv(filename, index=False)

    print(f"✓ Data saved to {filename}")
    print(f"✓ Dataset shape: {final_df.shape}")
    print(f"✓ Date range: {final_df['Date'].min().date()} to {final_df['Date'].max().date()}")
    print(f"✓ Total sales: ${final_df['Sales'].sum():,.2f}")
    print(f"✓ Average sales per transaction: ${final_df['Sales'].mean():.2f}")

    # Show sample data
    print("\nSample data:")
    print(final_df.head(10))

    return final_df


if __name__ == "__main__":
    # Generate sample data
    df = generate_retail_data("train.csv", 50000)
    print("\nSample retail data generated successfully!")
