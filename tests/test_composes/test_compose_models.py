import os
import pytest

from column_transformer import (
    column_transformer,
    complex_column_transformer,
    nested_column_transformer_with_pipeline,
  )

from transformed_target_regressor import (
    transformed_target_regressor,
    complex_transformed_target_regressor,
)

COMPOSE_MODEL_TESTS = [
    column_transformer,
    complex_column_transformer,
    transformed_target_regressor,
    complex_transformed_target_regressor,
    nested_column_transformer_with_pipeline,
]

@pytest.fixture(scope="session", autouse=True)
def reset_exported_models_directory():
    exported_models_directory = os.path.join(
        os.getcwd(), "tests", "exported_composes")
    if not os.path.isdir(exported_models_directory):
        os.mkdir(exported_models_directory)
        return
    for file_name in os.listdir(exported_models_directory):
        # construct full file path
        json_file = os.path.join(exported_models_directory, file_name)
        if os.path.isfile(json_file):
            os.remove(json_file)

def test_full():
    for model in COMPOSE_MODEL_TESTS:
        print("Testing model: ", model.__name__)
        model()
