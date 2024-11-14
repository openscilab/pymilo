# -*- coding: utf-8 -*-
"""PyMilo chain for cross decomposition models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..pymilo_param import SKLEARN_CROSS_DECOMPOSITION_TABLE
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes

from ..utils.util import get_sklearn_type

from traceback import format_exc

CROSS_DECOMPOSITION_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
}


def is_cross_decomposition(model):
    """
    Check if the input model is a sklearn's cross decomposition model.

    :param model: is a string name of a cross decomposition or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_CROSS_DECOMPOSITION_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_CROSS_DECOMPOSITION_TABLE.keys()


def transport_cross_decomposition(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given cross decomposition model to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn cross decomposition model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_cross_decomposition(request)
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
            return deserialize_cross_decomposition(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_cross_decomposition(cross_decomposition_object):
    """
    Return the serialized json string of the given cross decomposition model.

    :param cross_decomposition_object: given model to be get serialized
    :type cross_decomposition_object: any sklearn cross decomposition model
    :return: the serialized json string of the given cross decomposition model
    """
    for transporter in CROSS_DECOMPOSITION_CHAIN:
        CROSS_DECOMPOSITION_CHAIN[transporter].transport(
            cross_decomposition_object, Command.SERIALIZE)
    return cross_decomposition_object.__dict__


def deserialize_cross_decomposition(cross_decomposition, is_inner_model=False):
    """
    Return the associated sklearn cross decomposition model.

    :param cross_decomposition: given json string of a cross decomposition model to get deserialized to associated sklearn cross decomposition model
    :type cross_decomposition: obj
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn cross decomposition model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_CROSS_DECOMPOSITION_TABLE[cross_decomposition["type"]]()
        data = cross_decomposition["data"]
    else:
        raw_model = SKLEARN_CROSS_DECOMPOSITION_TABLE[cross_decomposition.type]()
        data = cross_decomposition.data

    for transporter in CROSS_DECOMPOSITION_CHAIN:
        CROSS_DECOMPOSITION_CHAIN[transporter].transport(
            cross_decomposition, Command.DESERIALIZE, is_inner_model)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn cross decomposition model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_cross_decomposition(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_cross_decomposition(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
