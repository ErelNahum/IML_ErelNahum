from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

PERCENTAGE_RANGE = (10, 100)

LOSS_MEAN_LENGTH = 10



def preprocess_data(X: pd.DataFrame, y: Optional[pd.Series] = None):
    """
    preprocess data
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector corresponding given samples

    Returns
    -------
    Post-processed design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    if isinstance(y, pd.Series): # Only for training data
        y = y.dropna()
        y = y[y > 0]
        X = X.loc[y.index]
        X = X.dropna()

    X = X.drop(columns=["id", "lat", "long", "date", "sqft_lot15", "sqft_living15", "zipcode"])

    if isinstance(y, pd.Series):
        for col in ["sqft_living", "sqft_lot", "sqft_above", "yr_built", "bathrooms", "floors", "sqft_basement", "yr_renovated"]:
            X = X[X[col] >= 0]
        y = y.loc[X.index]

    return (X, y) if isinstance(y, pd.Series) else X


def feature_evaluation(X: pd.DataFrame, y: pd.Series, output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    print('started features')
    for feature in X.columns:
        pearson_correlation = np.cov(X[feature], y)[0, 1] / (np.std(X[feature]) * np.std(y))
        print(pearson_correlation)
        px.scatter(x=X[feature], y=y, title=f"Price as function of {feature}. Pearson Correlation: {pearson_correlation}").update_layout(xaxis_title=feature, yaxis_title='price').write_html(output_path + f'/{feature}-{round(pearson_correlation, 2)}.html')

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv("../datasets/house_prices.csv")

    # Question 1 - split data into train and test sets
    train_X, train_y, test_X, test_y = split_train_test(df.drop(columns=['price']), df['price'])

    # Question 2 - Preprocessing of housing prices dataset
    train_X, train_y = preprocess_data(train_X, train_y)
    test_X = preprocess_data(test_X)

    # Question 3 - Feature evaluation with respect to response
    feature_evaluation(train_X, train_y, './features')

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    percentages = np.arange(PERCENTAGE_RANGE[0], PERCENTAGE_RANGE[1] + 1)
    loss_matrix = np.zeros((len(percentages), LOSS_MEAN_LENGTH))
    for percentage_index, percentage in enumerate(percentages):
        for j in range(LOSS_MEAN_LENGTH):
            sub_train_X = train_X.sample(frac=percentage / 100)
            sub_train_y = train_y.loc[sub_train_X.index]
            loss_matrix[percentage_index, j] = LinearRegression(True).fit(sub_train_X, sub_train_y).loss(test_X, test_y)

    MSE_mean = np.mean(loss_matrix, axis=1)
    MSE_std = np.std(loss_matrix, axis=1)

    go.Figure([
        go.Scatter(x=percentages, y=MSE_mean, mode='lines+markers', line=dict(color='blue'), name='Mean of MSE'),
        go.Scatter(x=percentages, y=MSE_mean - 2 * MSE_std, mode='lines', line=dict(color='gray'), showlegend=False),
        go.Scatter(x=percentages, y=MSE_mean + 2 * MSE_std, mode='lines', fill='tonexty', line=dict(color='gray'), showlegend=False)],
        layout=go.Layout(
            title="MSE as function of Training Data Size",
            xaxis=dict(title='Percentage'),
            yaxis=dict(title='MSE')))\
        .show()

