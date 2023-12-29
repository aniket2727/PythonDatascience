# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate some random data for demonstration
np.random.seed(42)
square_footage = np.random.randint(1000, 5000, 100)
house_price = 50000 + 150 * square_footage + np.random.normal(0, 10000, 100)

# Reshape the data
square_footage = square_footage.reshape(-1, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(square_footage, house_price, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f"Mean Squared Error: {mse}")

# Plot the data and the regression line
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
plt.xlabel('Square Footage')
plt.ylabel('House Price')
plt.title('Linear Regression Model')
plt.show()
