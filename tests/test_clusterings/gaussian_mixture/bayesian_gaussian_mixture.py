from sklearn.mixture import BayesianGaussianMixture

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Bayesian Gaussian Mixture"

def bayesian_gaussian_mixture():
    x, y = prepare_simple_clustering_datasets()
    bayesian_gaussian_mixture = BayesianGaussianMixture(n_components=2, random_state=42).fit(x, y)
    pymilo_clustering_test(bayesian_gaussian_mixture, MODEL_NAME, x, True)
