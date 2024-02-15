# tests.py
import numpy as np
from sklearn.linear_model import LinearRegression

def test_regression():
    # Generate some sample data
    X = np.array([[1], [2], [3], [4], [5]])
    y = np.array([2, 4, 6, 8, 10])

    # Create a linear regression model
    model = LinearRegression()

    # Fit the model to the data
    model.fit(X, y)

    # Predict using the trained model
    x_new = np.array([[6]])
    y_pred = model.predict(x_new)

    # Check the predicted value
    assert np.isclose(y_pred[0], 12.0), "Prediction error"
