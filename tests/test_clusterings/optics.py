from sklearn.cluster import OPTICS

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "OPTICS"

def optics():    
    x, y = prepare_simple_clustering_datasets()
    optics = OPTICS().fit(x, y)
    pymilo_clustering_test(optics, MODEL_NAME, x)
