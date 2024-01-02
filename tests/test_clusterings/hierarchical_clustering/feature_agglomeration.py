from sklearn.cluster import FeatureAgglomeration

from pymilo.utils.test_pymilo import pymilo_clustering_test
from pymilo.utils.data_exporter import prepare_simple_clustering_datasets

MODEL_NAME = "Feature Agglomeration"

def feature_agglomeration():    
    x, y = prepare_simple_clustering_datasets()
    feature_agglomeration = FeatureAgglomeration(n_clusters=2).fit(x, y)
    pymilo_clustering_test(feature_agglomeration, MODEL_NAME, x)
