import numpy as np


class LinearRegression:

    def __init__(self, x, y, learning_rate=0.001, num=1000):
        arr = np.ones(x.shape)
        x = np.append(arr, x, axis=1)
        self.x = x
        self.y = y
        self.row = x.shape[0]
        self.col = x.shape[1]
        self.learning_rate = learning_rate
        self.num_iterations = num
        self.params = np.random.randn(x.shape[1])
        # self.params = np.random.randn(x.shape[1])
        self.gradient_descent_main()
    # y = m * x + c

    def sum_of_squared_errors(self, y_test, y_pred):
        error = np.sum((y_test - y_pred) ** 2)
        return error / (2 * self.row)

    def gradient_descent_main(self):
        for i in range(self.num_iterations):
            predicted = np.matmul(self.x, self.params)
            error = predicted - self.y
            self.params -= ((self.learning_rate / self.row) * (self.x.T.dot(error)))

    def predict(self, x_test, y_test):
        arr = np.ones(x_test.shape)
        x_test = np.append(arr, x_test, axis=1)
        self.y_pred = np.matmul(x_test, self.params)
        error = self.sum_of_squared_errors(y_test, self.y_pred)

        return self.y_pred, error

    def getParams(self):
        return self.params

