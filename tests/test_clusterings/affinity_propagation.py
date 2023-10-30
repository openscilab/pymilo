from sklearn.cluster import AffinityPropagation

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Affinity Propagation"

def affinity_propagation():    
    x, y = prepare_simple_clustering_datasets()
    affinity_propagation = AffinityPropagation(random_state=5).fit(x, y)
    pymilo_clustering_test(affinity_propagation, MODEL_NAME, x)
