from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.offline.offline
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    raise NotImplementedError()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    raise NotImplementedError()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    raise NotImplementedError()


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 1 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    train_X, train_y = X[:n_samples], y[:n_samples]
    test_X, test_y = X[n_samples:], y[n_samples:]

    # Question 2 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_params = np.linspace(1e-5, 0.05, n_evaluations)
    lasso_params = np.linspace(1e-3, 2, n_evaluations)

    ridge_scores = np.zeros((n_evaluations, 2))
    lasso_scores = np.zeros((n_evaluations, 2))

    for i in range(n_evaluations):
        ridge_lambda = ridge_params[i]
        lasso_lambda = lasso_params[i]

        ridge_scores[i] = cross_validate(RidgeRegression(ridge_lambda), train_X, train_y, mean_square_error)
        lasso_scores[i] = cross_validate(Lasso(lasso_lambda, max_iter=3000), train_X, train_y, mean_square_error)
    q2_fig = make_subplots(1, 2, subplot_titles=[r"$\text{Ridge Regression}$", r"$\text{Lasso Regression}$"], shared_xaxes=True)\
        .update_layout(title=r"$\text{Train and Validation Errors (averaged over the k-folds)}$")\
        .update_xaxes(title=r"$\lambda\text{ - Regularization parameter}$")\
        .add_traces([go.Scatter(x=ridge_params, y=ridge_scores[:, 0], name="Ridge Train Error"),
                    go.Scatter(x=ridge_params, y=ridge_scores[:, 1], name="Ridge Validation Error"),
                    go.Scatter(x=lasso_params, y=lasso_scores[:, 0], name="Lasso Train Error"),
                    go.Scatter(x=lasso_params, y=lasso_scores[:, 1], name="Lasso Validation Error")],
                    rows=[1, 1, 1, 1],
                    cols=[1, 1, 2, 2])
    q2_fig.show()

    # Question 3 - Compare best Ridge model, best Lasso model and Least Squares model
    best_ridge_lambda = ridge_params[np.argmin(ridge_scores[:, 1])]
    best_lasso_lambda = ridge_params[np.argmin(lasso_scores[:, 1])]

    fitted_ridge_loss = RidgeRegression(best_ridge_lambda).fit(train_X, train_y).loss(test_X, test_y)
    fitted_lasso_loss = mean_square_error(Lasso(best_lasso_lambda).fit(train_X, train_y).predict(test_X), test_y)
    linear_reg_loss = LinearRegression().fit(train_X, train_y).loss(test_X, test_y)


    q3_fig = go.Figure(
        data=[
            go.Bar(x=['Ridge Loss', 'Lasso Loss', 'Linear Regression Loss'],
                   y=[fitted_ridge_loss, fitted_lasso_loss, linear_reg_loss])
        ],
        layout=go.Layout(title='Test Error Comparison')
    )
    q3_fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    select_regularization_parameter()
