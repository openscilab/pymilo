import os
import json

def write_and_read(serialized_model, file_addr):
    with open(file_addr, 'w') as fp:
        fp.write(json.dumps(serialized_model, indent=4))
    with open(file_addr, 'r') as fp:
        return json.load(fp)

def get_path(model_name, index=None):
    index = "" if index is None else f"_{index}"
    return  os.path.join(os.getcwd(), "tests", "exported_composes", model_name + index + ".json")
