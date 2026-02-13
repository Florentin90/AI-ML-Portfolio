import pandas as pd

# Load dataset
df = pd.read_csv("tickets.csv")

# Basic text cleaning
df["clean_text"] = df["text"].str.lower()

print("Preprocessing completed.")
print(df.head())
