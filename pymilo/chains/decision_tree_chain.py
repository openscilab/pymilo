# -*- coding: utf-8 -*-
"""PyMilo chain for decision trees."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.tree_transporter import TreeTransporter

from ..pymilo_param import SKLEARN_DECISION_TREE_TABLE

from ..exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeSerilaizatoinErrorTypes
from traceback import format_exc


DECISION_TREE_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
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
        return type(model) in SKLEARN_DECISION_TREE_TABLE.values()


def transport_decision_tree(request, command):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given decision tree model to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: the transported request as a json string or sklearn decision tree model
    """
    _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_decision_tree(request)
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
            return deserialize_decision_tree(request)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
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


def deserialize_decision_tree(decision_tree):
    """
    Return the associated sklearn decision tree model of the given decision_tree.

    :param decision_tree: given json string of a decision tree model to get deserialized to associated sklearn decision tree model
    :type decision_tree: obj
    :return: associated sklearn decision tree model
    """
    raw_model = SKLEARN_DECISION_TREE_TABLE[decision_tree.type]()
    data = decision_tree.data

    for transporter in DECISION_TREE_CHAIN:
        DECISION_TREE_CHAIN[transporter].transport(
            decision_tree, Command.DESERIALZIE)
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
                    'error_type': SerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALZIE:
        if is_decision_tree(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeSerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
