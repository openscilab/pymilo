import os
import pytest

from kmeans import kmeans
from affinity_propagation import affinity_propagation
from mean_shift import mean_shift
from dbscan import dbscan
from optics import optics
from spectral_clustering import spectral_clustering
from gaussian_mixture.gaussian_mixture import gaussian_mixture

CLUSTERINGS = {
    "KMEANS": [kmeans],
    "AFFINITY_PROPAGATION": [affinity_propagation],
    "MEAN_SHIFT": [mean_shift],
    "DBSCAN": [dbscan],
    "OPTICS": [optics],
    "SPECTRAL_CLUSTERING": [spectral_clustering],
    "GAUSSIAN_MIXTURE": [gaussian_mixture],
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
    for category in CLUSTERINGS.keys():
        for model in CLUSTERINGS[category]:
            model()
