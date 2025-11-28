import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

# Load data
df = pd.read_csv("Decision.csv")

if "Id" in df.columns:
    df=df.drop(columns=["Id"])

# Separate input and target
X = df.drop("Buys", axis=1)
y = df["Buys"]

# Create a label encoder for each column
encoders = {}
for col in X.columns:
    enc = LabelEncoder()
    X[col] = enc.fit_transform(X[col])
    encoders[col] = enc   # store encoder for later use

# Encode the target column
target_enc = LabelEncoder()
y = target_enc.fit_transform(y)

# Train decision tree
model = DecisionTreeClassifier()
model.fit(X, y)

# TEST sample
test = pd.DataFrame({
    "Age": ["<21"],
    "Income": ["High"],
    "Gender": ["Male"],
    "Ms": ["Single"]
})

# ENCODE using stored encoders (important!)
for col in test.columns:
    test[col] = encoders[col].transform(test[col])

# Predict
pred = model.predict(test)[0]

print("Encoded prediction:", pred)
print("Final prediction:", target_enc.inverse_transform([pred])[0])
