# -*- coding: utf-8 -*-
"""Functions."""
import numpy as np
import sklearn

from .chains.linear_model_chain import transport_linear_model, is_linear_model
from .chains.neural_network_chain import transport_neural_network, is_neural_network
from .chains.decision_tree_chain import transport_decision_tree, is_decision_tree
from .chains.clustering_chain import transport_clusterer, is_clusterer
from .chains.naive_bayes_chain import transport_naive_bayes, is_naive_bayes

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
    if is_linear_model(model):
        return transport_linear_model(model, Command.SERIALIZE)
    elif is_neural_network(model):
        return transport_neural_network(model, Command.SERIALIZE)
    elif is_decision_tree(model):
        return transport_decision_tree(model, Command.SERIALIZE)
    elif is_clusterer(model):
        return transport_clusterer(model, Command.SERIALIZE)
    elif is_naive_bayes(model):
        return transport_naive_bayes(model, Command.SERIALIZE)
    else:
        return None


def to_sklearn_model(import_obj):
    """
    Deserialize the imported object as a sklearn model.

    :param import_obj: given object
    :type import_obj: pymilo.Import
    :return: sklearn model
    """
    if is_linear_model(import_obj.type):
        return transport_linear_model(import_obj, Command.DESERIALZIE)
    elif is_neural_network(import_obj.type):
        return transport_neural_network(import_obj, Command.DESERIALZIE)
    elif is_decision_tree(import_obj.type):
        return transport_decision_tree(import_obj, Command.DESERIALZIE)
    elif is_clusterer(import_obj.type):
        return transport_clusterer(import_obj, Command.DESERIALZIE)
    elif is_naive_bayes(import_obj.type):
        return transport_naive_bayes(import_obj, Command.DESERIALZIE)
    else:
        return None


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
    if len(exported_output) != len(imported_output):
        return False  # TODO: throw exception
    total_error = 0
    for key in exported_output:
        if key not in imported_output:
            return False  # TODO: throw exception
        total_error += np.abs(imported_output[key] - exported_output[key])
    return np.abs(total_error) < epsilon_error
