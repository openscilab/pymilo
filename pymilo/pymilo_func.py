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


def convert_to_sklearn_model(import_obj):
    return transport_linear_model(import_obj, Command.DESERIALZIE)


def compare_model_outputs(exported_model_output,
                          imported_model_output,
                          epsilon_error=10**(-8)):
    if len(exported_model_output.keys()) != len(imported_model_output.keys()):
        return False  # Todo throw exception
    totalError = 0
    for key in exported_model_output.keys():
        if (not (key in imported_model_output.keys())):
            return False  # Todo throw exception
        totalError += np.abs(imported_model_output[key]) - \
            np.abs(exported_model_output[key])
    # print(f'totalError: {totalError}')
    return np.abs(totalError) < epsilon_error
