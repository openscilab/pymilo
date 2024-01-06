from sklearn.cluster import BisectingKMeans

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Bisecting KMeans"

def bisecting_kmeans():
    x, y = prepare_simple_clustering_datasets()
    bisecting_kmeans = BisectingKMeans(n_clusters=3, random_state=0).fit(x, y)
    pymilo_clustering_test(bisecting_kmeans, MODEL_NAME, x, True)
