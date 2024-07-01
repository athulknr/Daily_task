import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import LeaveOneOut

# Create the dataset
data = {
    'size': [1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200],
    'bedrooms': [3, 3, 3, 4, 4, 4, 5, 5],
    'price': [300000, 320000, 340000, 360000, 380000, 400000, 420000, 440000]
}
df = pd.DataFrame(data)

# Features and labels
X = df[['size', 'bedrooms']].values
y = df['price'].values

# Initialize the model
model = LinearRegression()

# Define the Leave-One-Out cross-validation method
loo = LeaveOneOut()

mae_scores = []

# Perform Leave-One-Out cross-validation
for train_index, test_index in loo.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)
    mae_scores.append(mae)
    print(f"Fold MAE: {mae:.2f}")

# Calculate the average MAE
average_mae = np.mean(mae_scores)
print(f"Mean Absolute Error: {average_mae:.2f}")
