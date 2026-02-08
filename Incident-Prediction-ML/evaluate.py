# evaluate.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("data.csv")

# Convert Severity to numeric
severity_map = {"Low": 1, "Medium": 2, "High": 3}
df['Severity_num'] = df['Severity'].map(severity_map)

# Features & target
X = df[['Severity_num']]
y = pd.factorize(df['IncidentType'])[0]

# Train the same model
model = DecisionTreeClassifier()
model.fit(X, y)

# Predict
predictions = model.predict(X)

# Evaluate
accuracy = accuracy_score(y, predictions)
print(f"Accuracy on training data: {accuracy:.2f}")
