import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib  # for saving the model

# Load dataset
df = pd.read_csv("tickets.csv")
df["clean_text"] = df["text"].str.lower()

# Vectorize text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["category"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model and vectorizer
joblib.dump(model, "ticket_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model training completed and saved.")
