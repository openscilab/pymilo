# -*- coding: utf-8 -*-
"""PyMilo chain for clustering models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.function_transporter import FunctionTransporter
from ..transporters.cfnode_transporter import CFNodeTransporter

from ..pymilo_param import SKLEARN_CLUSTERING_TABLE, NOT_SUPPORTED
from ..exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeSerilaizatoinErrorTypes
from traceback import format_exc

bisecting_kmeans_support = SKLEARN_CLUSTERING_TABLE["BisectingKMeans"] != NOT_SUPPORTED
CLUSTERING_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "FunctionTransporter": FunctionTransporter(),
    "CFNodeTransporter": CFNodeTransporter(),
}

if bisecting_kmeans_support:
    from ..transporters.randomstate_transporter import RandomStateTransporter
    from ..transporters.bisecting_tree_transporter import BisectingTreeTransporter
    CLUSTERING_CHAIN["RandomStateTransporter"] = RandomStateTransporter()
    CLUSTERING_CHAIN["BisectingTreeTransporter"] = BisectingTreeTransporter()


def is_clusterer(model):
    """
    Check if the input model is a sklearn's clustering model.

    :param model: is a string name of a clusterer or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_CLUSTERING_TABLE
    else:
        return type(model) in SKLEARN_CLUSTERING_TABLE.values()


def transport_clusterer(request, command):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given clusterer to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: the transported request as a json string or sklearn clustering model
    """
    _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_clusterer(request)
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
            return deserialize_clusterer(request)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_clusterer(clusterer_object):
    """
    Return the serialized json string of the given clustering model.

    :param clusterer_object: given model to be get serialized
    :type clusterer_object: any sklearn clustering model
    :return: the serialized json string of the given clusterer
    """
    for transporter in CLUSTERING_CHAIN:
        CLUSTERING_CHAIN[transporter].transport(
            clusterer_object, Command.SERIALIZE)
    return clusterer_object.__dict__


def deserialize_clusterer(clusterer):
    """
    Return the associated sklearn clustering model of the given clusterer.

    :param clusterer: given json string of a clustering model to get deserialized to associated sklearn clustering model
    :type clusterer: obj
    :return: associated sklearn clustering model
    """
    raw_model = SKLEARN_CLUSTERING_TABLE[clusterer.type]()
    data = clusterer.data

    for transporter in CLUSTERING_CHAIN:
        CLUSTERING_CHAIN[transporter].transport(
            clusterer, Command.DESERIALZIE)
    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn clusterer model or a json string of it, serialized through the pymilo export.
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_clusterer(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALZIE:
        if is_clusterer(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeSerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
