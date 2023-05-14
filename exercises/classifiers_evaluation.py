from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"), ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset('../datasets/' + f)

        # Fit Perceptron and record loss in each fit iteration
        losses = []

        def stage_loss(fitted_model, temp, temp2):
            losses.append(fitted_model.loss(X, y))

        Perceptron(callback=stage_loss).fit(X,y)

        # Plot figure of loss as function of fitting iteration
        fig = go.Figure(data=go.Scatter(x=list(range(len(losses))), y=losses, mode='lines+markers'),
                        layout=go.Layout(
                            xaxis_title= 'Iteration',
                            yaxis_title= 'Misclassification Error',
                            title= f'{n} Dataset - Misclassification Error as function of Iteration'
                        ))
        fig.show()


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines", marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset('../datasets/' + f)

        # Fit models and predict over training set
        naive_fitter = GaussianNaiveBayes().fit(X, y)
        lda_fitter = LDA().fit(X, y)

        naive_prediction = naive_fitter.predict(X)
        lda_prediction = lda_fitter.predict(X)

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(1, 2, subplot_titles=(f'Gaussian Naive Bayes({accuracy(y, naive_prediction)})', f'LDA({accuracy(y, lda_prediction)})'))

        # Add traces for data-points setting symbols and colors
        bayes_scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                   marker=dict(color=naive_prediction, symbol=class_symbols[y]))
        lda_scatter = go.Scatter(x=X[:, 0], y=X[:, 1], mode='markers',
                                 marker=dict(color=lda_prediction, symbol=class_symbols[y]))
        fig.add_traces([bayes_scatter, lda_scatter], rows=[1, 1], cols=[1, 2])

        # Add `X` dots specifying fitted Gaussians' means
        bayes_x = go.Scatter(x=naive_fitter.mu_[:, 0], y=naive_fitter.mu_[:, 1], mode='markers',
                             marker=dict(color='black', symbol='x', size=10))
        lda_x = go.Scatter(x=lda_fitter.mu_[:, 0], y=lda_fitter.mu_[:, 1], mode='markers',
                           marker=dict(color='black', symbol='x', size=10))
        fig.add_traces([bayes_x, lda_x], rows=[1, 1], cols=[1, 2])

        # Add ellipses depicting the covariances of the fitted Gaussians
        for i in range(len(naive_fitter.classes_)):
            fig.add_traces([get_ellipse(naive_fitter.mu_[i], np.diag(naive_fitter.vars_[i])),
                            get_ellipse(lda_fitter.mu_[i], lda_fitter.cov_)], rows=[1, 1], cols=[1, 2])

        fig.update_layout(title_text=f'Gaussian Classifiers Comparison ({f})', showlegend=False)
        fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
