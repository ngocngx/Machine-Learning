import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, degree=1, reg=None, tol=1e-6, alpha=0.0):
        self.degree = degree
        self.reg = reg
        self.alpha = alpha
        self.tol = tol
        self.w = None
    
    def fit(self, X, y):
        X_poly = self._generate_polynomial_features(X)
        if self.reg == 'l2':
            self.w = self._ridge_regression(X_poly, y)
        elif self.reg == 'l1':
            self.w = self._lasso_regression(X_poly, y)
        else:
            self.w = self._ordinary_least_squares(X_poly, y)
    
    def predict(self, X):
        X_poly = self._generate_polynomial_features(X)
        return X_poly.dot(self.w)
    
    def _generate_polynomial_features(self, X):
        return np.vander(X, self.degree + 1, increasing=True)
    
    def _ordinary_least_squares(self, X, y):
        w_ols = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y) 
        return w_ols
    
    def _ridge_regression(self, X, y):
        identity_matrix = np.identity(X.shape[1])
        w_ridge = np.linalg.inv(X.T.dot(X) + self.alpha * identity_matrix).dot(X.T).dot(y)
        return w_ridge
    
    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0
    
    def _coordinate_descent(self, X, y, w, alpha):
        n, m = X.shape
        for _ in range(1000):
            w_prev = w.copy()
            
            for j in range(m):
                # Compute residuals excluding feature j
                r = y - X.dot(w) + X[:, j] * w[j]
                
                # Soft thresholding
                z = np.dot(X[:, j], r)
                w[j] = self._soft_thresholding_operator(z, self.alpha) / (X[:, j]**2).sum()
                
            # Convergence check
            if np.sqrt(((w - w_prev)**2).sum()) < self.tol:
                break
                
        return w

    def _lasso_regression(self, X, y):
        w_initial = np.zeros(X.shape[1])
        w_lasso =  self._coordinate_descent(X, y, w_initial, self.alpha)
        return  w_lasso
    
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y - y_pred))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        r2_score = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
        print('MSE: {:.3f}, RMSE: {:.3f}, MAE: {:.3f}, R2 Score: {:.6f}%, MAPE: {:.6}%'.format(mse, rmse, mae, r2_score*100, mape))
    
    def plot_data(self, X, y):
        plt.figure(figsize=(8, 4))
        plt.scatter(X, y, s=60, facecolors='none', edgecolors='blue')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()
    
    def plot(self, X, y):
        X_pred = np.linspace(X.min(), X.max(), 100)
        y_pred = self.predict(X_pred)

        plt.figure(figsize=(8, 4))
        plt.scatter(X, y, s=60, facecolors='none', edgecolors='blue')
        plt.plot(X_pred, y_pred, color='red')
        plt.title('n = {}, degree = {}, reg = {}, α = {}'.format(X.shape[0], self.degree, self.reg, self.alpha))
        plt.xlabel('X')
        plt.ylabel('y')
        plt.show()


class LogisticRegression:
    def __init__(self, alpha=0.01, num_iter=1000, threshold=0.5, tol=1e-4):
        self.alpha = alpha
        self.num_iter = num_iter
        self.threshold = threshold
        self.w = None
        self.tol = tol

    def sigmoid(self, z):
        # Clip z to avoid overflow
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))
    
    def loss(self, y, y_pred):
        return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    
    def fit(self, X, y):
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])
        self.w = np.zeros(n + 1)
        for i in range(self.num_iter):
            z = np.dot(X, self.w)
            y_pred = self.sigmoid(z)
            grad = np.dot(X.T, (y_pred - y)) / m
            self.w -= self.alpha * grad
            if np.linalg.norm(grad) < self.tol:
                break
        return self.w
    
    def predict(self, X):
        m, n = X.shape
        X = np.hstack([np.ones((m, 1)), X])
        if self.w is None:
            raise Exception("Model not trained yet")
        z = np.dot(X, self.w)
        y_pred = self.sigmoid(z)
        return (y_pred >= self.threshold).astype(int)
    
    def score(self, X, y):
        return np.mean(self.predict(X) == y)
    
    def plot_decision_boundary(self, X, y):
        if X.shape[1] != 2:
            raise Exception("Can only plot decision boundary for 2D data")
        x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                               np.arange(x2_min, x2_max, 0.01))
        Z = self.predict(np.c_[xx1.ravel(), xx2.ravel()])
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.4)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        plt.show()


class NaiveBayes:
    def __init__(self) -> None:
        self.classes = []
        self.priors = []
        self.means = []
        self.vars = []
    
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.priors = np.array([np.mean(y == c) for c in self.classes])
        self.means = np.array([np.mean(X[y == c], axis=0) for c in self.classes])
        self.vars = np.array([np.var(X[y == c], axis=0) for c in self.classes])
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.priors[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self.means[class_idx]
        var = self.vars[class_idx]
        coeff = 1 / np.sqrt(2 * np.pi * var )
        exponent = np.exp(-(x - mean)**2 / (2 * var))
        return coeff * exponent
    
    def score(self, y_true, y_pred):
        return np.mean(y_true == y_pred)