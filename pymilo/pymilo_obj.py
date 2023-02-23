# -*- coding: utf-8 -*-
"""PyMilo modules."""
from pymilo.pymilo_func import get_sklearn_data, get_sklearn_version, to_sklearn_model
from pymilo.utils.util import get_sklearn_type
from pymilo.pymilo_param import PYMILO_VERSION
import json


class Export:
    """
    TODO: Complete docstring.
    """

    def __init__(self, model):
        self.data = get_sklearn_data(model)
        self.version = get_sklearn_version()
        self.type = get_sklearn_type(model)

    def save(self, file_adr):
        """
        Save model in a file.

        :param file_adr: file address
        :type file_adr: str
        :return: None
        """
        with open(file_adr, 'w') as fp:
            json.dump({
                "data": self.data,
                "sklearn_version": self.version,
                "pymilo_version": PYMILO_VERSION,
                "model_type": self.type
            }, fp, indent=4)

class Import:
    """
    TODO: Complete docstring.
    """

    def __init__(self, file_adr, json_dump=None):
        serialized_model_obj = None
        if json_dump and isinstance(json_dump, str):
            serialized_model_obj = json.loads(json_dump)
        else:
            with open(file_adr, 'r') as fp:
                serialized_model_obj = json.load(fp)
        self.data = serialized_model_obj["data"]
        self.version = serialized_model_obj["sklearn_version"]
        self.type = serialized_model_obj["model_type"]

    def to_model(self):
        """
        Convert imported model to sklearn model.

        :return: sklearn model
        """
        return to_sklearn_model(self)
