import numpy as np
from scipy.linalg import expm
import math

class Logistic:

	def __init__(self, learning_rate=0.001, num=1000):
		self.learning_rate = learning_rate
		self.num_iterations = num


	def fit(self, x, y):
		arr = np.ones(x.shape[0])
		x = np.append(arr.reshape(arr.shape[0],1), x, axis=1)
		self.x = x
		self.y = y
		self.row = x.shape[0]
		self.col = x.shape[1]
		self.params = np.random.randn(x.shape[1])
		self.gradient_descent_main()

# Hypothesis : h(x) = 1 / (1 + e ^ (- theta.T * x))
# Cost Function : J(theta) = - (np.sum(y * log(h(x)) + (1 - y) * log(1 - h(x)))) / m

	def get_dim(self):
		return self.row, self.col

	def cost_function(self, y_pred):
		first = np.matmul(self.y.reshape(self.y.shape[0],1), np.log(y_pred))
		h1 = 1 - self.y
		h2 = 1 - y_pred
		second = np.matmul(h1, np.log(h2))
		cost = first + second

		return  (- np.sum(cost) / self.row)

	def gradient_descent_main(self):
		for i in range(self.num_iterations):
			error = np.array([])
			for j in range(0, self.row):
				mult = np.matmul(self.params, self.x[j])
				sum_mult = np.sum(mult)
				sum_mult = np.negative(sum_mult)
				val = math.exp(sum_mult)
				hypothesis = 1 / (1 + val)
				error = np.append(error, hypothesis - self.y[j])

			self.params -= ((self.learning_rate / self.row) * (self.x.T.dot(error)))


	def predict(self, x_test):
		hypothesis = np.array([])
		arr = np.ones(x_test.shape[0])
		x_test = np.append(arr.reshape(arr.shape[0],1), x_test, axis=1)
		for i in range(0, len(x_test)):
			mult = np.matmul(self.params, x_test[i])
			sum_mult = np.sum(mult)
			sum_mult = np.negative(sum_mult)
			val = math.exp(sum_mult)
			h = 1 / (1 + val)
			hypothesis = np.append(hypothesis, h)	


		for k in range(0, len(hypothesis)):
			if hypothesis[k] < 0.5:
				hypothesis[k] = 0
			else:
				hypothesis[k] = 1
		return hypothesis


	def score(self, y_test, y_pred):
		
		wrong_cnt = 0
		right_cnt = 0
		for i in range(len(y_test)):
			if(y_test[i] == y_pred[i]):
				right_cnt += 1
			else:
				wrong_cnt += 1

		return (right_cnt/(right_cnt+wrong_cnt) * 100)


	def get_params(self):
		return self.params







