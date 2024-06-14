import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#dataset
height =np.array([150,160,164,165,173]).reshape(-1,1)
weight = np.array([50,65,63,68,72])

#create a linear regression model for the above dataset
model =LinearRegression()

#lets fit the model with appropriate date

model.fit(height,weight)

predicted_weight = model.predict(height)


print(f"intercept: {model.intercept_}")
print(f"coeffiecents: {model.coef_[0]}")

#create a scatterplot for the above
plt.scatter(height, weight, color='blue', label='Actual weights')
plt.plot(height, predicted_weight, color='red', label='Fitted line')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Height vs Weight Linear Regression')
plt.legend()
plt.show()