from sklearn.cluster import MiniBatchKMeans

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "MiniBatch KMeans"

def minibatch_kmeans():
    x, y = prepare_simple_clustering_datasets()
    minibatch_kmeans = MiniBatchKMeans(n_clusters=2, random_state=2, batch_size=6, max_iter=10, n_init="auto").fit(x, y)
    pymilo_clustering_test(minibatch_kmeans, MODEL_NAME, x, True)
