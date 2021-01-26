import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Import dataset using pandas. Can be modified
dataset = pd.read_csv('sample.csv')

# Use commands below to 1) show first 5 entries and 2) show characteristics like mean, median, etc of dataset
# print(dataset.head())
# print(dataset.describe())

# Columns of CSV required for prediction
X = dataset[['Water_Temperature','Transducer_Depth','Wave_Period']]

# Column of CSV to be predicted
y = dataset['Wave_Height']

# The below command sets 80% of data to training and 20% to test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Since we're using linear regression:
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Since multiple variables are available to perform prediction, the coeffecient for each column
# determines the perfect variable chosen to perform prediction.
coeff_df = pd.DataFrame(regressor.coef_, X.columns, columns=['Coefficient'])
print(coeff_df)

# To make predictions on the test data, execute the following script:
y_pred = regressor.predict(X_test)

# # To compare the actual output values for X_test with the predicted values, execute the following script:
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df)

# The metrics below show the deviation between the predicted and actual values
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))