# -*- coding: utf-8 -*-
"""PyMilo chain for linear models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.sgdoptimizer_transporter import SGDOptimizerTransporter
from ..transporters.adamoptimizer_transporter import AdamOptimizerTransporter
from ..transporters.labelbinarizer_transporter import LabelBinarizerTransporter

from ..pymilo_param import SKLEARN_NEURAL_NETWORK_TABLE

from ..exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeSerilaizatoinErrorTypes
from traceback import format_exc


NEURAL_NETWORK_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
    "SGDOptimizer": SGDOptimizerTransporter(),
    "AdamOptimizerTransporter": AdamOptimizerTransporter(),
    "LabelBinarizerTransporter": LabelBinarizerTransporter(),
}


def is_neural_network(model):
    """
    Check if the input model is a sklearn's neural network.

    :param model: is a string name of a neural network or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_NEURAL_NETWORK_TABLE
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
    _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_neural_network(request)
        except Exception as e:
            raise PymiloSerializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc(),
                    },
                    'object': request,
                })

    elif command == Command.DESERIALZIE:
        try:
            return deserialize_neural_network(request)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_neural_network(neural_network_object):
    """
    Return the serialized json string of the given neural network model.

    :param neural_network_object: given model to be get serialized
    :type neural_network_object: any sklearn neural network model
    :return: the serialized json string of the given neural network model
    """
    for transporter in NEURAL_NETWORK_CHAIN:
        NEURAL_NETWORK_CHAIN[transporter].transport(
            neural_network_object, Command.SERIALIZE)
    return neural_network_object.__dict__


def deserialize_neural_network(neural_network):
    """
    Return the associated sklearn neural network model of the given neural_network.

    :param neural_network: given json string of a neural network model to get deserialized to associated sklearn NN model
    :type neural_network: obj
    :return: associated sklearn NN model
    """
    raw_model = SKLEARN_NEURAL_NETWORK_TABLE[neural_network.type]()
    data = neural_network.data

    for transporter in NEURAL_NETWORK_CHAIN:
        NEURAL_NETWORK_CHAIN[transporter].transport(
            neural_network, Command.DESERIALZIE)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn neural network model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_neural_network(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALZIE:
        if is_neural_network(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeSerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
