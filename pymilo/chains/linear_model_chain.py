# -*- coding: utf-8 -*-
"""PyMilo chain for linear models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.baseloss_transporter import BaseLossTransporter
from ..transporters.lossfunction_transporter import LossFunctionTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..pymilo_param import SKLEARN_LINEAR_MODEL_TABLE
from ..utils.util import get_sklearn_type, is_iterable

from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes
from traceback import format_exc


LINEAR_MODEL_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "BaseLossTransporter": BaseLossTransporter(),
    "LossFunctionTransporter": LossFunctionTransporter(),
}


def is_linear_model(model):
    """
    Check if the input model is a sklearn's linear model.

    :param model: name of a linear model or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_LINEAR_MODEL_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_LINEAR_MODEL_TABLE.keys()


def is_deserialized_linear_model(content):
    """
    Check if the given content is a previously serialized model by Pymilo's Export or not.

    :param content: given object to be authorized as a valid pymilo exported serialized model
    :type content: any object
    :return: check result as bool
    """
    if not is_iterable(content):
        return False
    return "pymilo-inner-model-type" in content and "pymilo-inner-model-data" in content


def transport_linear_model(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given model to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn model
    """
    if not is_inner_model:
        validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_linear_model(request)
        except Exception as e:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})

    elif command == Command.DESERIALIZE:
        try:
            return deserialize_linear_model(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_linear_model(linear_model_object):
    """
    Return the serialized json string of the given linear model.

    :param linear_model_object: given model to be get serialized
    :type linear_model_object: any sklearn linear model
    :return: the serialized json string of the given linear model
    """
    # first serializing the inner linear models...
    for key in linear_model_object.__dict__:
        if is_linear_model(linear_model_object.__dict__[key]):
            linear_model_object.__dict__[key] = {
                "pymilo-inner-model-data": transport_linear_model(linear_model_object.__dict__[key], Command.SERIALIZE, True),
                "pymilo-inner-model-type": get_sklearn_type(linear_model_object.__dict__[key]),
                "pymilo-bypass": True
            }
    # now serializing non-linear model fields
    for transporter in LINEAR_MODEL_CHAIN:
        LINEAR_MODEL_CHAIN[transporter].transport(
            linear_model_object, Command.SERIALIZE)
    return linear_model_object.__dict__


def deserialize_linear_model(linear_model, is_inner_model):
    """
    Return the associated sklearn linear model of the given linear_model.

    :param linear_model: given json string of a linear model to get deserialized to associated sklearn linear model
    :type linear_model: obj
    :param is_inner_model: determines whether it is an inner model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn linear model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_LINEAR_MODEL_TABLE[linear_model["type"]]()
        data = linear_model["data"]
    else:
        raw_model = SKLEARN_LINEAR_MODEL_TABLE[linear_model.type]()
        data = linear_model.data
    # first deserializing the inner linear models(one depth inner linear
    # models have been deserialized -> TODO full depth).
    for key in data:
        if is_deserialized_linear_model(data[key]):
            data[key] = transport_linear_model({
                "data": data[key]["pymilo-inner-model-data"],
                "type": data[key]["pymilo-inner-model-type"]
            }, Command.DESERIALIZE, is_inner_model=True)
    # now deserializing non-linear models fields
    for transporter in LINEAR_MODEL_CHAIN:
        LINEAR_MODEL_CHAIN[transporter].transport(
            linear_model, Command.DESERIALIZE, is_inner_model)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn linear model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_linear_model(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        model_type = model.type
        if is_linear_model(model_type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
