import os
import pytest

from kneighbors_regressor import kneighbors_regressor
from kneighbors_classifier import kneighbors_classifier
from radius_neighbors_regressor import radius_neighbors_regressor
from radius_neighbors_classifier import radius_neighbors_classifier
from nearest_neighbor import nearest_neighbor
from nearest_centroid import nearest_centroid
from local_outlier_factor import local_outlier_factor

NEIGHBORS = {
    "KNeighbors": [kneighbors_regressor, kneighbors_classifier],
    "RadiusNeighbors": [radius_neighbors_regressor, radius_neighbors_classifier],
    "Nearests": [nearest_neighbor, nearest_centroid],
    "LocalOutlierDetectors": [local_outlier_factor],
}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_neighbors")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in NEIGHBORS:
        for model in NEIGHBORS[category]:
            model()
