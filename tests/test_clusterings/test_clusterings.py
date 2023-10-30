import os
import pytest

from kmeans import kmeans
from affinity_propagation import affinity_propagation

CLUSTERINGS = {
    "KMEANS": [kmeans],
    "AFFINITY_PROPAGATION": [affinity_propagation],
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
