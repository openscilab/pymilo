# -*- coding: utf-8 -*-
"""Functions."""
import numpy as np
import sklearn
from .chains.linear_model_chain import transport_linear_model
from .transporters.transporter import Command


def get_sklearn_version():
    """
    Return sklearn version.

    :return: sklearn version as a str
    """
    return sklearn.__version__


def get_sklearn_data(model):
    """
    Return sklearn data by serializing given model.

    :param model: given model
    :type model: any sklearn's model class
    :return: sklearn data
    """
    return transport_linear_model(model, Command.SERIALIZE)


def to_sklearn_model(import_obj):
    """
    Deserialize the imported object as a sklearn model.

    :param import_obj: given object
    :type import_obj: pymilo.Import
    :return: sklearn model
    """
    return transport_linear_model(import_obj, Command.DESERIALZIE)


def compare_model_outputs(exported_output,
                          imported_output,
                          epsilon_error=10**(-8)):
    """
    Check if the given models outputs are the same.

    :param exported_output: exported model output
    :type exported_output: dict
    :param imported_output: imported model output
    :type imported_output: dict
    :param epsilon_error: error threshold for numeric comparisons
    :type epsilon_error: float
    :return: check result as bool
    """
    if len(exported_output.keys()) != len(imported_output.keys()):
        return False  # TODO: throw exception
    total_error = 0
    for key in exported_output.keys():
        if not key in imported_output.keys():
            return False  # TODO: throw exception
        total_error += np.abs(imported_output[key] - exported_output[key])
    return np.abs(total_error) < epsilon_error
