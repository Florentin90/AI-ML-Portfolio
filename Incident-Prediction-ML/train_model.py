# train_model.py
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load dataset
df = pd.read_csv("data.csv")

# Convert Severity to numeric for simple model
severity_map = {"Low": 1, "Medium": 2, "High": 3}
df['Severity_num'] = df['Severity'].map(severity_map)

# Features: use numeric Severity (toy example)
X = df[['Severity_num']]
# Target: IncidentType (encode as numbers)
y = pd.factorize(df['IncidentType'])[0]

# Train a simple Decision Tree
model = DecisionTreeClassifier()
model.fit(X, y)

# Show predictions on training data
predictions = model.predict(X)
print("Predictions on training data:", predictions)
