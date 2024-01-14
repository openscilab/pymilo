from sklearn.cluster import SpectralBiclustering

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Spectral Biclustering"

def spectral_biclustering():
    x, y = prepare_simple_clustering_datasets()
    spectral_biclustering = SpectralBiclustering(n_clusters=2, random_state=0).fit(x, y)
    pymilo_clustering_test(spectral_biclustering, MODEL_NAME, x)
