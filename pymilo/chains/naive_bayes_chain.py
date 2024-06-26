# -*- coding: utf-8 -*-
"""PyMilo chain for naive bayes models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..pymilo_param import SKLEARN_NAIVE_BAYES_TABLE
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes

from ..utils.util import get_sklearn_type

from traceback import format_exc

NAIVE_BAYES_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
}


def is_naive_bayes(model):
    """
    Check if the input model is a sklearn's naive bayes model.

    :param model: is a string name of a naive bayes or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_NAIVE_BAYES_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_NAIVE_BAYES_TABLE.keys()


def transport_naive_bayes(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given naive bayes to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn naive bayes model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_naive_bayes(request)
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
            return deserialize_naive_bayes(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_naive_bayes(naive_bayes_object):
    """
    Return the serialized json string of the given naive bayes model.

    :param naive_bayes_object: given model to be get serialized
    :type naive_bayes_object: any sklearn naive bayes model
    :return: the serialized json string of the given naive bayes
    """
    for transporter in NAIVE_BAYES_CHAIN:
        NAIVE_BAYES_CHAIN[transporter].transport(
            naive_bayes_object, Command.SERIALIZE)
    return naive_bayes_object.__dict__


def deserialize_naive_bayes(naive_bayes, is_inner_model=False):
    """
    Return the associated sklearn naive bayes model of the given naive bayes.

    :param naive bayes: given json string of a naive bayes model to get deserialized to associated sklearn naive bayes model
    :type naive bayes: obj
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn naive bayes model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_NAIVE_BAYES_TABLE[naive_bayes["type"]]()
        data = naive_bayes["data"]
    else:
        raw_model = SKLEARN_NAIVE_BAYES_TABLE[naive_bayes.type]()
        data = naive_bayes.data

    for transporter in NAIVE_BAYES_CHAIN:
        NAIVE_BAYES_CHAIN[transporter].transport(
            naive_bayes, Command.DESERIALIZE, is_inner_model)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn naive bayes model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_naive_bayes(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_naive_bayes(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
