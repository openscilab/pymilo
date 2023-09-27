from sklearn.cluster import KMeans

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Kmeans"

def kmeans():    
    x, y = prepare_simple_clustering_datasets()
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x, y)
    pymilo_clustering_test(kmeans, MODEL_NAME, x)
