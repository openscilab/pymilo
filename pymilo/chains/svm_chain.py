# -*- coding: utf-8 -*-
"""PyMilo chain for svm models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..pymilo_param import SKLEARN_SVM_TABLE
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes

from ..utils.util import get_sklearn_type

from traceback import format_exc

SVM_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
}


def is_svm(model):
    """
    Check if the input model is a sklearn's svm model.

    :param model: is a string name of a svm or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_SVM_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_SVM_TABLE.keys()


def transport_svm(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given svm to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn svm model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_svm(request)
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
            return deserialize_svm(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_svm(svm_object):
    """
    Return the serialized json string of the given svm model.

    :param svm_object: given model to be get serialized
    :type svm_object: any sklearn svm model
    :return: the serialized json string of the given svm
    """
    for transporter in SVM_CHAIN:
        SVM_CHAIN[transporter].transport(
            svm_object, Command.SERIALIZE)
    return svm_object.__dict__


def deserialize_svm(svm, is_inner_model=False):
    """
    Return the associated sklearn svm model of the given svm.

    :param svm: given json string of a svm model to get deserialized to associated sklearn svm model
    :type svm: obj
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn svm model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_SVM_TABLE[svm["type"]]()
        data = svm["data"]
    else:
        raw_model = SKLEARN_SVM_TABLE[svm.type]()
        data = svm.data

    for transporter in SVM_CHAIN:
        SVM_CHAIN[transporter].transport(
            svm, Command.DESERIALIZE, is_inner_model)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn svm model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_svm(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_svm(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
