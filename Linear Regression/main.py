import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from LinearRegression import LinearRegression

data = pd.read_csv('Salary_Data.csv')
X = data.iloc[:, 0:1].values
y = data.iloc[:, 1].values

sc = MinMaxScaler()
X = sc.fit_transform(X)
print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=45)

print(X_train.shape)

# Applying Linear Regression
regressor = LinearRegression(X_train, Y_train)
print(regressor.getParams())
y_predicted, error = regressor.predict(X_test, Y_test)

# print(y_predicted, error)

