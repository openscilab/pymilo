# -*- coding: utf-8 -*-
"""PyMilo chain for clustering models."""
from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.function_transporter import FunctionTransporter
from ..transporters.cfnode_transporter import CFNodeTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..utils.util import get_sklearn_type

from ..pymilo_param import SKLEARN_CLUSTERING_TABLE, NOT_SUPPORTED
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes
from traceback import format_exc

bisecting_kmeans_support = SKLEARN_CLUSTERING_TABLE["BisectingKMeans"] != NOT_SUPPORTED
CLUSTERING_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
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
        return get_sklearn_type(model) in SKLEARN_CLUSTERING_TABLE.keys()


def transport_clusterer(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given clusterer to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn clustering model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_clusterer(request)
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
            return deserialize_clusterer(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
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


def deserialize_clusterer(clusterer, is_inner_model=False):
    """
    Return the associated sklearn clustering model of the given clusterer.

    :param clusterer: given json string of a clustering model to get deserialized to associated sklearn clustering model
    :type clusterer: obj
    :param is_inner_model: determines whether it is an inner linear model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn clustering model
    """
    raw_model = None
    data = None
    if is_inner_model:
        raw_model = SKLEARN_CLUSTERING_TABLE[clusterer["type"]]()
        data = clusterer["data"]
    else:
        raw_model = SKLEARN_CLUSTERING_TABLE[clusterer.type]()
        data = clusterer.data

    for transporter in CLUSTERING_CHAIN:
        CLUSTERING_CHAIN[transporter].transport(
            clusterer, Command.DESERIALIZE, is_inner_model)
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
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_clusterer(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
