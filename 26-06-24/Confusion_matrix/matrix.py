import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Read the CSV file
df = pd.read_csv('work.csv')

# Convert categorical data to numeric
df['gender'] = df['gender'].map({'male': 1, 'female': 0})
df['smoking'] = df['smoking'].map({'yes': 1, 'no': 0})
df['diabetes'] = df['diabetes'].map({'yes': 1, 'no': 0})
df['exercise'] = df['exercise'].map({'yes': 1, 'no': 0})

# Check the balance of the target variable
print("Target variable distribution:\n", df['heart_attack'].value_counts())

# Features and target variable
X = df.drop('heart_attack', axis=1)
y = df['heart_attack']

# Use StratifiedKFold for cross-validation
skf = StratifiedKFold(n_splits=3)

# Logistic Regression model
model = LogisticRegression()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

# Display the cross-validation scores
print("Cross-validation scores:", scores)
print("Mean accuracy:", scores.mean())

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)

# Logistic Regression model
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Output metrics
print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)