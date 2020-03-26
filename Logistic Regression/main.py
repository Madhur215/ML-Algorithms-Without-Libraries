import numpy as np
import pandas as pd
from logisticClassifier import Logistic
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Social_Network_Ads.csv')
data = data.drop(['User ID'], axis=1)
# print(data)

data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1}).astype(int)

X = data.iloc[:, 0:3].values
y = data.iloc[:, 3].values

sc = StandardScaler()
X = sc.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=45)
regressor = Logistic()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(y_pred)
print(regressor.get_dim())
print(regressor.get_params().shape)

print(regressor.score(y_test, y_pred))


