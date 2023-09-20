import os
import pytest

from decision_tree.decision_tree_regression import decision_tree_regression
from decision_tree.decision_tree_classification import decision_tree_classification
DECISION_TREES = {
    "DECISION_TREE": [decision_tree_regression],

}

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_decision_trees")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for category in DECISION_TREES.keys():
        for model in DECISION_TREES[category]:
            model()
