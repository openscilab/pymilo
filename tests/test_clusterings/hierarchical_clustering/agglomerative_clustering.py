from sklearn.cluster import AgglomerativeClustering

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Agglomerative Clustering"

def agglomerative_clustering():
    x, y = prepare_simple_clustering_datasets()
    agglomerative_clustering = AgglomerativeClustering().fit(x, y)
    pymilo_clustering_test(agglomerative_clustering, MODEL_NAME, x)
