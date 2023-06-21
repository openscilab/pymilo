# CORRUPTED_JSON_FILE = 1 -> tested.
# INVALID_MODEL = 2 -> tested.
# VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3 -> tested.
from pymilo.pymilo_obj import Import

import os

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
 