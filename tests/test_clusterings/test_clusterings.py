import os
import pytest

from kmeans import kmeans
from affinity_propagation import affinity_propagation
from mean_shift import mean_shift
from dbscan import dbscan

try:
    from hdbscan import hdbscan
except BaseException:
    print("HDBSCAN doesn't exist in this version of python.")

from optics import optics
from spectral_clustering import spectral_clustering
from gaussian_mixture.gaussian_mixture import gaussian_mixture
from hierarchical_clustering.agglomerative_clustering import agglomerative_clustering
from hierarchical_clustering.feature_agglomeration import feature_agglomeration
from pymilo.pymilo_param import SKLEARN_CLUSTERING_TABLE, NOT_SUPPORTED

CLUSTERINGS = {
    "KMEANS": [kmeans],
    "AFFINITY_PROPAGATION": [affinity_propagation],
    "MEAN_SHIFT": [mean_shift],
    "DBSCAN": [dbscan, hdbscan if SKLEARN_CLUSTERING_TABLE["HDBSCAN"] != NOT_SUPPORTED else (None,"HDBSCAN")],
    "OPTICS": [optics],
    "SPECTRAL_CLUSTERING": [spectral_clustering],
    "GAUSSIAN_MIXTURE": [gaussian_mixture],
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
