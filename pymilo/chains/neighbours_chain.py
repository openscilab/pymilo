# -*- coding: utf-8 -*-
"""PyMilo chain for neighbors models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.neighbors_tree_transporter import NeighborsTreeTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..pymilo_param import SKLEARN_NEIGHBORS_TABLE
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes

from ..utils.util import get_sklearn_type

from traceback import format_exc

NEIGHBORS_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "NeighborsTreeTransporter": NeighborsTreeTransporter(),
}


def is_neighbors(model):
    """
    Check if the input model is a sklearn's neighbors model.

    :param model: is a string name of a neighbor or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_NEIGHBORS_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_NEIGHBORS_TABLE.keys()


def transport_neighbor(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given neighbor to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn neighbors model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_neighbor(request)
        except Exception as e:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc(),
                    },
                    'object': request,
                })

    elif command == Command.DESERIALIZE:
        try:
            return deserialize_neighbor(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_neighbor(neighbor_object):
    """
    Return the serialized json string of the given neighbor model.

    :param neighbor_object: given model to be get serialized
    :type neighbor_object: any sklearn neighbor model
    :return: the serialized json string of the given neighbor
    """
    for transporter in NEIGHBORS_CHAIN:
        NEIGHBORS_CHAIN[transporter].transport(
            neighbor_object, Command.SERIALIZE)
    return neighbor_object.__dict__


def deserialize_neighbor(neighbor, is_inner_model=False):
    """
    Return the associated sklearn neighbor model of the given neighbor.

    :param neighbor: given json string of a neighbor model to get deserialized to associated sklearn neighbors model
    :type neighbor: obj
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn neighbor model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_NEIGHBORS_TABLE[neighbor["type"]]()
        data = neighbor["data"]
    else:
        raw_model = SKLEARN_NEIGHBORS_TABLE[neighbor.type]()
        data = neighbor.data

    for transporter in NEIGHBORS_CHAIN:
        NEIGHBORS_CHAIN[transporter].transport(
            neighbor, Command.DESERIALIZE, is_inner_model)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn neighbor model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_neighbors(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_neighbors(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
