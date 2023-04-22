import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df.drop_duplicates().dropna()
    df = df[df.Temp > 0]
    df["DayOfYear"] = df['Date'].dt.dayofyear
    df["Year"] = df["Year"].astype(str)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    df = load_data('../datasets/City_Temperature.csv')

    # Question 2 - Exploring data for specific country
    israel_df = df.loc[df.Country == 'Israel']

    px.scatter(israel_df, x="DayOfYear",
               y="Temp", color="Year",
               title="Average Temparture in Israel as function of Day of Year")\
        .show()

    px.bar(israel_df.groupby(['Month'], as_index=False)
           .agg(sigma=('Temp', 'std')),
           x='Month', y='sigma',
           title="Standard Deviation as function of Month - in Israel")\
        .show()

    # Question 3 - Exploring differences between countries
    px.line(df.groupby(['Country', 'Month'], as_index=False)
            .agg(sigma=('Temp', 'std'), mean=('Temp', 'mean')),
            x='Month', y='mean', error_y='sigma', color='Country')\
        .show()
    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(israel_df.DayOfYear, israel_df.Temp)
    polynom_degrees = np.arange(1, 11)
    errors = []
    for k in polynom_degrees:
        error = PolynomialFitting(k).fit(train_X.to_numpy(), train_y.to_numpy()).loss(test_X.to_numpy(), test_y.to_numpy())
        errors.append(np.round(error, 2))
    print(list(zip(polynom_degrees, errors)))
    px.bar(x=polynom_degrees, y=errors, text=errors,
           labels={'x': 'Fitted Polynom Degree', 'y': 'MSE over Test Set'})\
        .show()

    # Question 5 - Evaluating fitted model on different countries
    israel_model = PolynomialFitting(5).fit(israel_df.DayOfYear.to_numpy(), israel_df.Temp.to_numpy())
    countries = ['Jordan', 'South Africa', 'The Netherlands']
    errors = []
    for country in countries:
        country_df = df[df.Country == country]
        error = israel_model.loss(country_df.DayOfYear.to_numpy(), country_df.Temp.to_numpy())
        errors.append(np.round(error, 2))
    px.bar(x=countries, y=errors, labels={'x': 'Country', 'y': 'MSE'}, title="").show()