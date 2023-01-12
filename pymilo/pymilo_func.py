# -*- coding: utf-8 -*-
"""Functions."""
import numpy as np
import sklearn
from .chains.linear_model_chain import transport_linear_model
from .transporters.transporter import Command


def get_sklearn_version():
    return sklearn.__version__


def get_sklearn_data(model):
    return transport_linear_model(model, Command.SERIALIZE)


def to_sklearn_model(import_obj):
    return transport_linear_model(import_obj, Command.DESERIALZIE)


def compare_model_outputs(exported_output,
                          imported_output,
                          epsilon_error=10**(-8)):
    if len(exported_output.keys()) != len(imported_output.keys()):
        return False  # TODO: throw exception
    total_error = 0
    for key in exported_output.keys():
        if (not (key in imported_output.keys())):
            return False  # TODO: throw exception
        total_error += np.abs(imported_output[key]) - \
            np.abs(exported_output[key])
    return np.abs(total_error) < epsilon_error
