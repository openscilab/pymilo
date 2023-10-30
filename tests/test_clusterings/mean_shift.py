from sklearn.cluster import MeanShift

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Mean Shift"

def mean_shift():    
    x, y = prepare_simple_clustering_datasets()
    mean_shift = MeanShift(bandwidth=2).fit(x, y)
    pymilo_clustering_test(mean_shift, MODEL_NAME, x)
