from sklearn.cluster import SpectralClustering

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Spectral Clustering"

def spectral_clustering():
    x, y = prepare_simple_clustering_datasets()
    spectral_clustering = SpectralClustering(random_state=5).fit(x, y)
    pymilo_clustering_test(spectral_clustering, MODEL_NAME, x)
