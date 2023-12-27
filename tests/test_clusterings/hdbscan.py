from sklearn.cluster import HDBSCAN

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "HDBSCAN"

def hdbscan():
    x, y = prepare_simple_clustering_datasets()
    hdbscan = HDBSCAN(min_cluster_size=20).fit(x, y)
    pymilo_clustering_test(hdbscan, MODEL_NAME, x)
