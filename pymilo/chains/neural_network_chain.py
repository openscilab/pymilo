# -*- coding: utf-8 -*-
"""PyMilo chain for linear models."""
from ..pymilo_param import SKLEARN_NEURAL_NETWORK_TABLE

def is_neural_network(model):
    """
    Check if the input model is a sklearn's neural network.

    :param model: is a string name of a neural network or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_NEURAL_NETWORK_TABLE.keys()
    else:
        return type(model) in SKLEARN_NEURAL_NETWORK_TABLE.values()

def transport_neural_network(request, command):
    """
    Return the transported (Serialized or Deserialized) model.
    
    :param request: given neural network model to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: the transported request as a json string or sklearn neural network model
    """
    return 
