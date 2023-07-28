import os
import pytest

from mlp.mlp_regression import multi_layer_perceptron_regression
    
NEURAL_NETWORKS = {
    "MLP_REGRESSION": [multi_layer_perceptron_regression],
}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_neural_networks")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in NEURAL_NETWORKS.keys():
        for model in NEURAL_NETWORKS[category]:
            model()
