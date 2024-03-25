# -*- coding: utf-8 -*-
"""PyMilo chain for neighbors models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.neighbors_tree_transporter import NeighborsTreeTransporter

from ..pymilo_param import SKLEARN_NEIGHBORS_TABLE
from ..exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeSerilaizatoinErrorTypes
from traceback import format_exc

NEIGHBORS_CHAIN = {
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
        return type(model) in SKLEARN_NEIGHBORS_TABLE.values()


def transport_neighbor(request, command):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given neighbor to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: the transported request as a json string or sklearn neighbors model
    """
    _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_neighbor(request)
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
            return deserialize_neighbor(request)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
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


def deserialize_neighbor(neighbor):
    """
    Return the associated sklearn neighbor model of the given neighbor.

    :param neighbor: given json string of a neighbor model to get deserialized to associated sklearn neighbors model
    :type neighbor: obj
    :return: associated sklearn neighbor model
    """
    raw_model = SKLEARN_NEIGHBORS_TABLE[neighbor.type]()
    data = neighbor.data

    for transporter in NEIGHBORS_CHAIN:
        NEIGHBORS_CHAIN[transporter].transport(
            neighbor, Command.DESERIALZIE)
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
                    'error_type': SerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALZIE:
        if is_neighbors(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeSerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
