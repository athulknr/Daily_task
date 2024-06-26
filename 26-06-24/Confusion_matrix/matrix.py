import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Sample dataset
data = {
    'Age': [55, 60, 45, 50, 65],
    'Gender': ['male', 'female', 'male', 'female', 'male'],
    'Cholesterol': [220, 180, 190, 200, 230],
    'BP': [140, 130, 110, 120, 150],
    'Smoking': ['yes', 'no', 'yes', 'no', 'yes'],
    'Diabetes': ['no', 'yes', 'no', 'no', 'yes'],
    'Exercise': ['yes', 'no', 'yes', 'yes', 'no'],
    'Heart_Attack': [1, 1, 0, 0, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Encode categorical variables
df['Gender'] = df['Gender'].map({'male': 0, 'female': 1})
df['Smoking'] = df['Smoking'].map({'yes': 1, 'no': 0})
df['Diabetes'] = df['Diabetes'].map({'yes': 1, 'no': 0})
df['Exercise'] = df['Exercise'].map({'yes': 1, 'no': 0})

# Features and target
X = df.drop('Heart_Attack', axis=1)
y = df['Heart_Attack']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train the logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test data
y_pred = model.predict(X_test)

# Evaluate the model
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print results
print("Confusion Matrix:")

print(conf_matrix)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
