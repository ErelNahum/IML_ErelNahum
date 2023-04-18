from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    mu = 10
    var = 1
    n_samples = 1000
    X = np.random.normal(mu, np.sqrt(var), n_samples)
    f = UnivariateGaussian().fit(X)
    print(f'({f.mu_}, {f.var_})')

    # Question 2 - Empirically showing sample mean is consistent
    errors = []
    n_samples = list(range(10, 1010, 10))
    for n in n_samples:
        estimated_mu = UnivariateGaussian().fit(X[:n])
        errors.append(np.abs(estimated_mu.mu_ - mu))
    go.Figure(go.Scatter(x=n_samples, y=errors, mode='markers'), layout=dict(template='plotly_white', title="Error of Sample Mean Estimator as a function of Number of Samples", xaxis_title="Number of Samples", yaxis_title=r"$|\hat\mu-\mu|$")).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    go.Figure(go.Scatter(x=X, y=f.pdf(X), mode='markers'), layout=dict(template='plotly_white', title="Empirical PDF of fitted model", xaxis_title="x", yaxis_title=r"$f_{\mu,\sigma^2}(x)$")).show()



def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    mu = np.array([0, 0, 4, 0])
    cov = np.array([[1, 0.2, 0, 0.5], [0.2, 2, 0, 0], [0, 0, 1, 0], [0.5, 0, 0, 1]])
    n_samples = 1000
    X = np.random.multivariate_normal(mu, cov, n_samples)
    f = MultivariateGaussian().fit(X)
    print(f.mu_)
    print(f.cov_)


    # Question 5 - Likelihood evaluation
    f1vals = np.linspace(-10, 10, 200)
    f3vals = np.linspace(-10, 10, 200)
    log_likelihood = np.zeros((200, 200))
    for i, f1 in enumerate(f1vals):
        for j, f3 in enumerate(f3vals):
            log_likelihood[i, j] = f.log_likelihood(np.array([f1, 0, f3, 0]), cov, X)

    go.Figure(go.Heatmap(x=f3vals, y=f1vals, z=log_likelihood), layout=dict(xaxis_title=r"$f_3$", yaxis_title=r"$f_1$", title=r"Log Likelihood for mu = [f1, 0, f3, 0] as functions of f1, f3")).show()

    # Question 6 - Maximum likelihood
    i, j = np.unravel_index(log_likelihood.argmax(), log_likelihood.shape)
    print((np.round(f1vals[i], 3), np.round(f3vals[j], 3)))


if __name__ == '__main__':
    np.random.seed(0)
    # test_univariate_gaussian()
    test_multivariate_gaussian()
