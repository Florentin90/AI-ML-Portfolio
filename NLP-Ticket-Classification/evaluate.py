import pandas as pd
from sklearn.metrics import accuracy_score
import joblib  # for loading the saved model

# Load dataset
df = pd.read_csv("tickets.csv")
df["clean_text"] = df["text"].str.lower()
X_text = df["clean_text"]
y = df["category"]

# Load saved model and vectorizer
model = joblib.load("ticket_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Transform text
X = vectorizer.transform(X_text)

# Predict
predictions = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, predictions)
print(f"Model Accuracy: {accuracy:.2f}")
