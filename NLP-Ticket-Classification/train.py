import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("tickets.csv")

# Convert text to lowercase
df["clean_text"] = df["text"].str.lower()

# Convert text into numerical features
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])

# Encode labels
y = df["category"]

# Train model
model = LogisticRegression()
model.fit(X, y)

print("Model training completed.")
