import os
import pytest

from pymilo.pymilo_param import SKLEARN_CLUSTERING_TABLE, NOT_SUPPORTED

from kmeans import kmeans
from affinity_propagation import affinity_propagation
from mean_shift import mean_shift
from dbscan import dbscan
from optics import optics
from spectral_clustering import spectral_clustering
from gaussian_mixture.gaussian_mixture import gaussian_mixture
from gaussian_mixture.bayesian_gaussian_mixture import bayesian_gaussian_mixture
from hierarchical_clustering.agglomerative_clustering import agglomerative_clustering
from hierarchical_clustering.feature_agglomeration import feature_agglomeration

bisecting_kmeans_support = True if SKLEARN_CLUSTERING_TABLE["BisectingKMeans"] != NOT_SUPPORTED else False
if bisecting_kmeans_support:
    from bisecting_kmeans import bisecting_kmeans

hdbscan_support = True if SKLEARN_CLUSTERING_TABLE["HDBSCAN"] != NOT_SUPPORTED else False
if hdbscan_support:
    from hdbscan import hdbscan

CLUSTERINGS = {
    "KMEANS": [kmeans, bisecting_kmeans if bisecting_kmeans_support else (None,"BisectingKMeans")],
    "AFFINITY_PROPAGATION": [affinity_propagation],
    "MEAN_SHIFT": [mean_shift],
    "DBSCAN": [dbscan, hdbscan if hdbscan_support else (None,"HDBSCAN")],
    "OPTICS": [optics],
    "SPECTRAL_CLUSTERING": [spectral_clustering],
    "GAUSSIAN_MIXTURE": [gaussian_mixture, bayesian_gaussian_mixture],
    "HIERARCHICAL_CLUSTERING": [agglomerative_clustering, feature_agglomeration],
}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_clusterings")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in CLUSTERINGS:
        for model in CLUSTERINGS[category]:
            if isinstance(model, tuple):
                func, model_name = model
                if func == None:
                    print("Model: " + model_name + " is not supported in this python version.")
                    continue
            model()
