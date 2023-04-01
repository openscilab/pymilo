# CORRUPTED_JSON_FILE = 1 -> tested.
# INVALID_MODEL = 2 -> tested.
# VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3
# UNKNOWN = 4

from pymilo.pymilo_obj import Import
import os

def test_invalid_json():
    json_files = ["corrupted", "unknown-model"]
    for json_file in json_files:
      json_path = os.path.join(os.getcwd(), "tests", "test_exceptions", "invalid_jsons", json_file + '.json')

      try:
        imported_model = Import(json_path)
        imported_model.to_model()
      except Exception as e:
         print("An Exception occured\n", e)
        