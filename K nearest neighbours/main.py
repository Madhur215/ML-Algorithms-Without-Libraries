import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from KNN import KNN

data = pd.read_csv('Social_Network_Ads.csv')
# print(data)

data = data.drop(['User ID'], axis=1)
data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1}).astype(int)
# print(data)

X = data.iloc[:, 0:3].values
Y = data.iloc[:, 3].values
# print(X)
# print(Y)

sc = StandardScaler()
X = sc.fit_transform(X)


X_train, X_test, Y_train ,Y_test = train_test_split(X, Y, test_size=0.15, random_state=99)
classifier = KNN(k=5)
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)

print(y_pred)

score = classifier.score(Y_test, y_pred)

print(score)