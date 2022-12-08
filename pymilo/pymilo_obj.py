# -*- coding: utf-8 -*-
"""PyMilo modules."""
from .pymilo_func import get_sklearn_data, get_sklearn_version, get_sklearn_type, convert_to_sklearn_model
from .pymilo_param import PYMILO_VERSION
import json

class Export:
    def __init__(self, model):
        self.data = get_sklearn_data(model)
        self.version = get_sklearn_version()
        self.type = get_sklearn_type(model)

    def to_json(self):
        return json.dumps(self.data)

    def save(self, file_adr):
        with open(file_adr, 'w') as fp:
            json.dump({"data": self.data, "sklearn_version": self.version, "pymilo_version": PYMILO_VERSION, "model_type": self.type}, fp)



class Import:
    def __init__(self, file_adr):
        with open(file_adr, 'r') as fp:
            file_data = json.load(fp)
        self.data = file_data["data"]
        self.version = file_data["sklearn_version"]
        self.type = file_data["model_type"]

    def to_model(self):
        return convert_to_sklearn_model(self)




