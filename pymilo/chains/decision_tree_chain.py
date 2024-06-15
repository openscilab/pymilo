# -*- coding: utf-8 -*-
"""PyMilo chain for decision trees."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.tree_transporter import TreeTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..utils.util import get_sklearn_type

from ..pymilo_param import SKLEARN_DECISION_TREE_TABLE

from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes
from traceback import format_exc


DECISION_TREE_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
    "TreeTransporter": TreeTransporter(),
}


def is_decision_tree(model):
    """
    Check if the input model is a sklearn's decision tree.

    :param model: is a string name of a decision tree or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_DECISION_TREE_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_DECISION_TREE_TABLE.keys()


def transport_decision_tree(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given decision tree model to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn decision tree model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_decision_tree(request)
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
            return deserialize_decision_tree(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_decision_tree(decision_tree_object):
    """
    Return the serialized json string of the given decision tree model.

    :param decision_tree_object: given model to be get serialized
    :type decision_tree_object: any sklearn decision tree model
    :return: the serialized json string of the given decision tree model
    """
    for transporter in DECISION_TREE_CHAIN:
        DECISION_TREE_CHAIN[transporter].transport(
            decision_tree_object, Command.SERIALIZE)
    return decision_tree_object.__dict__


def deserialize_decision_tree(decision_tree, is_inner_model=False):
    """
    Return the associated sklearn decision tree model of the given decision_tree.

    :param decision_tree: given json string of a decision tree model to get deserialized to associated sklearn decision tree model
    :type decision_tree: obj
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn decision tree model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_DECISION_TREE_TABLE[decision_tree["type"]]()
        data = decision_tree["data"]
    else:
        raw_model = SKLEARN_DECISION_TREE_TABLE[decision_tree.type]()
        data = decision_tree.data

    for transporter in DECISION_TREE_CHAIN:
        DECISION_TREE_CHAIN[transporter].transport(
            decision_tree, Command.DESERIALIZE, is_inner_model)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn decision tree model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_decision_tree(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_decision_tree(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
