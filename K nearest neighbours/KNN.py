import numpy as np
import pandas as pd


class KNN:

	def __init__(self, k=7):
		self.k = k

	def fit(self, X_train, Y_train):
		self.X_train = X_train
		self.Y_train = Y_train



	def predict(self, X_test):
		self.y_pred = np.array([])
		
		for x in X_test:
			dist = np.sum((x-self.X_train)**2, axis=1)
			dist = dist.reshape(dist.shape[0], 1)
			self.Y_train = self.Y_train.reshape(self.Y_train.shape[0], 1)
			distances = np.concatenate((dist,
			 			self.Y_train), axis=1)
			# distances = distances.argsort()
			distances = distances[distances[:, 0].argsort()]
			neighbours = distances[0:self.k]
			classes, counts = np.unique(neighbours[:,1], return_counts=True)
			self.y_pred = np.append(self.y_pred, classes[np.argmax(counts)])

		return self.y_pred


	def score(self, y_test, y_pred):

		wrong_cnt = 0
		right_cnt = 0
		for i in range(len(y_test)):
			if(y_test[i] == y_pred[i]):
				right_cnt += 1
			else:
				wrong_cnt += 1

		return (right_cnt/(right_cnt+wrong_cnt) * 100)