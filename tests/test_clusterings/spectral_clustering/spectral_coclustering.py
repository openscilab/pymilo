from sklearn.cluster import SpectralCoclustering

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Spectral Coclustering"

def spectral_coclustering():
    x, y = prepare_simple_clustering_datasets()
    spectral_coclustering = SpectralCoclustering(n_clusters=2, random_state=0).fit(x, y)
    pymilo_clustering_test(spectral_coclustering, MODEL_NAME, x)
