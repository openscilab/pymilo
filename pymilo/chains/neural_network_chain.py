# -*- coding: utf-8 -*-
"""PyMilo chain for linear models."""
from ..pymilo_param import SKLEARN_NEURAL_NETWORK_TABLE

def is_neural_network(model):
    if(isinstance(model, str)):
        return model in SKLEARN_NEURAL_NETWORK_TABLE.keys()
    else:
        return type(model) in SKLEARN_NEURAL_NETWORK_TABLE.values()

def transport_neural_network(request, command):
    return 
