import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("tickets.csv")

# Preprocess
df["clean_text"] = df["text"].str.lower()

# Vectorize
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["clean_text"])
y = df["category"]

# Train model
model = LogisticRegression()
model.fit(X, y)

# Predict
predictions = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, predictions)

print(f"Model Accuracy: {accuracy:.2f}")
