# INVALID_MODEL = 1 -> done.
# VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 2
# UNKNOWN = 3

from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import test_pymilo_regression
from pymilo.pymilo_obj import Import

import numpy as np

import os

# Learning model, but an invalid one.
# test case for INVALID_MODEL.
class InvalidModel:
  def __init__(self):
    self.name = "Invalid Linear Model"  

  def fit(self, x, y):
     return 

  def predict(self, x):
     return [0 for _ in range(np.shape(x)[0])]

def test_invalid_model():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create linear regression object
    model = InvalidModel()
    # Train the model using the training sets
    model.fit(x_train, y_train)
    return test_pymilo_regression(
        model, model.name , (x_test, y_test))

def test_invalid_json():
    json_files = ["invalid", "unknown-model"]
    for json_file in json_files:
      json_path = os.path.join(os.getcwd(), "tests", "test_exceptions", "invalid_jsons", json_file + '.json')

      try:
        imported_model = Import(json_path)
        imported_model.to_model()
      except Exception as e:
         print("An Exception occured\n", e)
        