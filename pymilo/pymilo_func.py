# -*- coding: utf-8 -*-
"""Functions."""
from .pymilo_param import SKLEARN_MODEL_TABLE
import numpy as np
import sklearn


def get_sklearn_data(model):
    data = model.__dict__
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    return data

def get_sklearn_version():
    return sklearn.__version__

def get_sklearn_type(model):
    raw_type = type(model)
    return str(raw_type).split(".")[-1][:-2]


def convert_to_sklearn_model(import_obj):
    raw_model = SKLEARN_MODEL_TABLE[import_obj.type]()
    data = import_obj.data
    for key in data.keys():
        if isinstance(data[key], list):
            data[key] = np.array(data[key])
    for item in data.keys():
        setattr(raw_model, item, data[item])
    return raw_model
