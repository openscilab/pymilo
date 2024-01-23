# -*- coding: utf-8 -*-
"""PyMilo chain for naive bayes models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter

from ..pymilo_param import SKLEARN_NAIVE_BAYES_TABLE
from ..exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeSerilaizatoinErrorTypes
from traceback import format_exc

NAIVE_BAYES_CHAIN = {
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
        return type(model) in SKLEARN_NAIVE_BAYES_TABLE.values()


def transport_naive_bayes(request, command):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given naive bayes to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: the transported request as a json string or sklearn naive bayes model
    """
    _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_naive_bayes(request)
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
            return deserialize_naive_bayes(request)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
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


def deserialize_naive_bayes(naive_bayes):
    """
    Return the associated sklearn naive bayes model of the given naive bayes.

    :param naive bayes: given json string of a naive bayes model to get deserialized to associated sklearn naive bayes model
    :type naive bayes: obj
    :return: associated sklearn naive bayes model
    """
    raw_model = SKLEARN_NAIVE_BAYES_TABLE[naive_bayes.type]()
    data = naive_bayes.data

    for transporter in NAIVE_BAYES_CHAIN:
        NAIVE_BAYES_CHAIN[transporter].transport(
            naive_bayes, Command.DESERIALZIE)
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
                    'error_type': SerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALZIE:
        if is_naive_bayes(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeSerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
