# CORRUPTED_JSON_FILE = 1 -> tested.
# INVALID_MODEL = 2 -> tested.
# VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3 -> tested.
import os
from pymilo.pymilo_obj import Import


def invalid_json(print_output = True):
    json_files = ["corrupted", "unknown-model"]
    for json_file in json_files:
      json_path = os.path.join(os.getcwd(), "tests", "test_exceptions", "invalid_jsons", json_file + '.json')
      try:
        imported_model = Import(json_path)
        imported_model.to_model()
        return False
      except Exception as e:
        if print_output: print("An Exception occured\n", e)
        return True
 
def invalid_url():
  url = "https://invalid_url"
  Import(url=url)

def valid_url_invalid_file():
  url = "https://filesamples.com/samples/code/json/sample1.json"
  Import(url=url)

def valid_url_valid_file():
  url = "https://raw.githubusercontent.com/openscilab/pymilo/main/tests/test_exceptions/valid_jsons/linear_regression.json"
  Import(url=url)