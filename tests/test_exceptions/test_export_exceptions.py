# INVALID_MODEL = 1 -> tested.
# VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 2 -> tested.
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from pymilo.utils.test_pymilo import pymilo_regression_test

from sklearn.linear_model import LinearRegression
from custom_models import CustomizedTweedieDistribution

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

def test_invalid_model(print_output = False):
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create linear regression object
    model = InvalidModel()
    # Train the model using the training sets
    model.fit(x_train, y_train)
    try:
      pymilo_regression_test(
        model, model.name , (x_test, y_test))
      return False
    except Exception as e:        
      if print_output: print("An Exception occured\n", e)
      return True

def test_valid_model_invalid_structure(print_output = False):
    MODEL_NAME = "LinearRegression"
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    # Create linear regression object
    linear_regression = LinearRegression()
    linear_regression.__dict__["invalid_field"] = CustomizedTweedieDistribution(power= 1.5)
    # Train the model using the training sets
    linear_regression.fit(x_train, y_train)
    try:
      pymilo_regression_test(
        linear_regression, MODEL_NAME, (x_test, y_test))
      return False
    except Exception as e:        
      if print_output: print("An Exception occured\n", e)
      return True
