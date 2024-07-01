import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer

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

# Discretize the target variable into bins
# Using 3 bins for this example
binner = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
y_binned = binner.fit_transform(y.reshape(-1, 1)).astype(int).ravel()

print("Binned y values:", y_binned)

# Initialize the model
model = LinearRegression()

# Define the stratified cross-validation method
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1)  # Reduced to 2 splits

mae_scores = []

# Perform Stratified K-Fold cross-validation
for train_index, test_index in skf.split(X, y_binned):
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
