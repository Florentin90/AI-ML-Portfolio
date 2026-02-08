# preprocessing.py
import pandas as pd

# Load the dummy dataset
df = pd.read_csv("data.csv")

# Show first 5 rows
print("First 5 rows of the dataset:")
print(df.head())
