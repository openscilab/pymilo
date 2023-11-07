from sklearn.mixture import GaussianMixture

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Gaussian Mixture"

def gaussian_mixture():    
    x, y = prepare_simple_clustering_datasets()
    gaussian_mixture = GaussianMixture(n_components=2, random_state=0).fit(x, y)
    pymilo_clustering_test(gaussian_mixture, MODEL_NAME, x)
