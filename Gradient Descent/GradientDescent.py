import numpy as np
import pandas as pd


class GradientDescent:

    def __int__(self, points, learning_rate=0.001, m_initial=0, c_initial=0, num=1000):
        self.learning_rate = learning_rate
        self.m_gradient = m_initial
        self.c_gradient = c_initial
        self.points = points
        self.number_iterations = num

    def sum_of_squared_errors(self):
        error = 0
        for i in range(len(self.points)):
            X = self.points[i, 0]
            Y = self.points[i, 1]
            error += (Y - (self.m_gradient * X + self.c_gradient)) ** 2

        return error / float(len(self.points))

    def gradient_descent_main(self):
        m = self.m_gradient
        c = self.c_gradient
        for i in range(self.number_iterations):
            m_grad = 0
            c_grad = 0
            n = float(len(self.points))
            for j in range(len(self.points)):
                x = self.points[i, 0]
                y = self.points[i, 1]
                m_grad += -(2 / n) * x * (y - (m * x + c))
                c_grad += -(2 / n) * (y - (m * x + c))

            m -= (self.learning_rate * m_grad)
            c -= (self.learning_rate * c_grad)

        return [m, c]

