import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    n_learners_options = list(range(1, n_learners + 1))
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)

    train_losses = [model.partial_loss(train_X, train_y, t) for t in n_learners_options]
    test_losses = [model.partial_loss(test_X, test_y, t) for t in n_learners_options]

    q1_fig = go.Figure(
        data=[
            go.Scatter(x=n_learners_options, y=train_losses, name='Train Errors', mode='lines'),
            go.Scatter(x=n_learners_options, y=test_losses, name='Test Errors', mode='lines')
        ],
        layout=go.Layout(
            xaxis_title=r'$\text{Number of Fitted Learners}$',
            yaxis_title=r'$\text{Misclassification Error (Normalized)}$',
            title=rf'$\text{{AdaBoost Misclassification As Function Of Number Of Classifiers}}, Noise={noise}$'
        )
    )
    q1_fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    q2_fig = make_subplots(rows=2, cols=2, subplot_titles=[rf"$\text{{{t} Classifiers}}, Noise={noise}$" for t in T])

    for i in range(len(T)):
        q2_fig.add_traces(
            [decision_surface(lambda X: model.partial_predict(X, T[i]), lims[0], lims[1]),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', marker=dict(color=test_y))
             ],
            rows=int(i/2)+1, cols=i % 2 + 1
        )
    q2_fig.show()

    # Question 3: Decision surface of best performing ensemble
    optimal_t = np.argmin(test_losses) + 1
    q3_fig = go.Figure(
        data=[
            decision_surface(lambda X: model.partial_predict(X, optimal_t), lims[0], lims[1]),
            go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode='markers', marker=dict(color=test_y))
        ],
        layout=go.Layout(
            title=f"Best Ensemble with {optimal_t} iterations. Accuracy - {1-test_losses[optimal_t - 1]}. Noise={noise}"
        )
    )
    q3_fig.show()

    # Question 4: Decision surface with weighted samples
    D = 15 * model.D_ / model.D_.max()
    q4_fig = go.Figure([
        decision_surface(model.predict, lims[0], lims[1]),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", marker=dict(
            size=D, color=train_y))],
        layout=go.Layout(title=f"Final AdaBoost Result including Weights, noise={noise}"))
    q4_fig.show()



if __name__ == '__main__':
    np.random.seed(0)
    for noise in [0, 0.4]:
        fit_and_evaluate_adaboost(noise)
