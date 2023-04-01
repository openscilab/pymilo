# INVALID_MODEL = 1 -> tested.
# VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 2
# UNKNOWN = 3

from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import test_pymilo_regression
import numpy as np

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
