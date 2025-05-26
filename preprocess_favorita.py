import pandas as pd

# --- Configuration ---
# Adjust this path if your downloaded Favorita train.csv is in a different location
FAVORITA_TRAIN_PATH = "train.csv"  # Assumes Favorita's train.csv is in the same folder
OUTPUT_PROJECT_TRAIN_PATH = "train.csv"  # This will be the train.csv your project uses

print(f"Loading Favorita dataset from: {FAVORITA_TRAIN_PATH}")

try:
    # Load the original training data from the Favorita dataset
    df_favorita = pd.read_csv(FAVORITA_TRAIN_PATH, parse_dates=["date"])
except FileNotFoundError:
    print(f"ERROR: File not found at {FAVORITA_TRAIN_PATH}")
    print(
        "Please make sure the Favorita train.csv is in the correct location and the path is correct."
    )
    exit()
except Exception as e:
    print(f"Error loading Favorita train.csv: {e}")
    exit()

print("Favorita dataset loaded successfully. Processing...")
print(f"Original Favorita data has {len(df_favorita)} rows.")

# --- Data Selection and Renaming ---
# Our scripts expect: 'Date', 'Store', 'Item', 'Category', 'Promotion', 'Sales'

# Rename columns from Favorita to match our project's expected names
df_processed = df_favorita.rename(
    columns={
        "date": "Date",  # Favorita 'date' -> Project 'Date'
        "store_nbr": "Store",  # Favorita 'store_nbr' -> Project 'Store'
        "family": "Item",  # Favorita 'family' (product category) will act as 'Item' for grouping
        # This is the most granular product identifier in Favorita's train.csv sales data
        "sales": "Sales",  # Favorita 'sales' -> Project 'Sales'
        "onpromotion": "Promotion",  # Favorita 'onpromotion' -> Project 'Promotion'
    }
)

# Create the 'Category' column. For simplicity, we'll use the 'family' (now 'Item') as 'Category' too.
# In a more advanced setup, you might map 'family' to a broader 'class' from items.csv.
df_processed["Category"] = df_processed["Item"]

# Select only the columns our project needs, in the desired order
# (Though order doesn't strictly matter for pandas, it's good for clarity)
project_columns = ["Date", "Store", "Item", "Category", "Promotion", "Sales"]
df_final = df_processed[project_columns]

# Convert Promotion to integer (0 or 1) if it's not already
# In Favorita, 'onpromotion' is usually count of items on promotion;
# let's convert it to a binary 0/1 for simplicity with existing scripts
# if it's just a count. Your scripts expect 0 or 1.
# A simple way is to check if it's > 0
df_final["Promotion"] = df_final["Promotion"].apply(lambda x: 1 if x > 0 else 0)


# --- Optional: Sampling (IMPORTANT FOR LARGE DATASET) ---
# The Favorita dataset is very large. For initial testing, you might want to use a sample.
# If you want to use the full dataset, comment out or delete the sampling lines.
# Example: Use data for only a few stores or a specific date range or a random sample.

# Option 1: Random sample of the data (e.g., 10% of the rows)
# df_final = df_final.sample(frac=0.1, random_state=42)
# print(f"Using a random sample of {len(df_final)} rows for faster processing.")

# Option 2: Filter for a specific date range (e.g., last year of data)
# df_final = df_final[df_final['Date'] >= '2017-01-01']
# print(f"Filtered data from 2017 onwards, resulting in {len(df_final)} rows.")

# Option 3: Filter for a few stores (e.g., stores 1 to 5)
# stores_to_sample = [1, 2, 3, 4, 5]
# df_final = df_final[df_final['Store'].isin(stores_to_sample)]
# print(f"Filtered for stores {stores_to_sample}, resulting in {len(df_final)} rows.")

print(
    f"Processed data has {len(df_final)} rows and columns: {df_final.columns.tolist()}"
)
print("\nFirst 5 rows of the processed data:")
print(df_final.head())

# --- Save the processed data ---
try:
    df_final.to_csv(OUTPUT_PROJECT_TRAIN_PATH, index=False)
    print(
        f"\nSuccessfully processed and saved the data to: {OUTPUT_PROJECT_TRAIN_PATH}"
    )
    print("This file will now be used by your main analysis and dashboard scripts.")
except Exception as e:
    print(f"\nError saving the processed file: {e}")
