# -*- coding: utf-8 -*-
"""useful utilities for chains."""

from .linear_model_chain import linear_chain
from .neural_network_chain import neural_network_chain
from .decision_tree_chain import decision_trees_chain
from .clustering_chain import clustering_chain
from .naive_bayes_chain import naive_bayes_chain
from .svm_chain import svm_chain
from .neighbours_chain import neighbors_chain
from .cross_decomposition_chain import cross_decomposition_chain
from ..utils.util import get_sklearn_type, check_str_in_iterable


MODEL_TYPE_TRANSPORTER = {
    "LINEAR_MODEL": linear_chain.transport,
    "NEURAL_NETWORK": neural_network_chain.transport,
    "DECISION_TREE": decision_trees_chain.transport,
    "CLUSTERING": clustering_chain.transport,
    "NAIVE_BAYES": naive_bayes_chain.transport,
    "SVM": svm_chain.transport,
    "NEIGHBORS": neighbors_chain.transport,
    "CROSS_DECOMPOSITION": cross_decomposition_chain.transport,
}


def get_concrete_transporter(model):
    """
    Get associated transporter for the given concrete(not ensemble) ML model.

    :param model: given model to get it's transporter
    :type model: scikit ML model
    :return: model category and transporter function
    """
    if isinstance(model, str):
        upper_model = model.upper()
        if upper_model in MODEL_TYPE_TRANSPORTER.keys():
            return upper_model, MODEL_TYPE_TRANSPORTER[upper_model]

    if linear_chain.is_supported(model):
        return "LINEAR_MODEL", linear_chain.transport
    elif neural_network_chain.is_supported(model):
        return "NEURAL_NETWORK", neural_network_chain.transport
    elif decision_trees_chain.is_supported(model):
        return "DECISION_TREE", decision_trees_chain.transport
    elif clustering_chain.is_supported(model):
        return "CLUSTERING", clustering_chain.transport
    elif naive_bayes_chain.is_supported(model):
        return "NAIVE_BAYES", naive_bayes_chain.transport
    elif svm_chain.is_supported(model):
        return "SVM", svm_chain.transport
    elif neighbors_chain.is_supported(model):
        return "NEIGHBORS", neighbors_chain.transport
    elif cross_decomposition_chain.is_supported(model):
        return "CROSS_DECOMPOSITION", cross_decomposition_chain.transport
    else:
        return None, None


def get_transporter(model):
    """
    Get associated transporter for the given ML model.

    :param model: given model to get it's transporter
    :type model: scikit ML model or str
    :return: model category and transporter function
    """
    # String routing
    if isinstance(model, str):
        upper = model.upper()
        if upper == "COMPOSE":
            from .compose_chain import compose_chain
            return "COMPOSE", compose_chain.transport
        if upper == "ENSEMBLE":
            from .ensemble_chain import ensemble_chain
            return "ENSEMBLE", ensemble_chain.transport

    # Object routing (check higher-level categories first)
    from .compose_chain import compose_chain
    if compose_chain.is_supported(model):
        return "COMPOSE", compose_chain.transport

    from .ensemble_chain import ensemble_chain
    if ensemble_chain.is_supported(model):
        return "ENSEMBLE", ensemble_chain.transport

    return get_concrete_transporter(model)


def serialize_possible_ml_model(model):
    """
    Serialize the given object if it is a supported ML model.

    :param model: given object
    :type model: any
    :return: ML model flag and serialized result
    """
    if isinstance(model, str):
        return False, model
    ml_category, transporter = get_transporter(model)
    if transporter is None:
        return False, model
    from ..transporters.transporter import Command
    return True, {
        "pymilo-bypass": True,
        "pymilo-inner-model-data": transporter(model, Command.SERIALIZE),
        "pymilo-inner-model-type": get_sklearn_type(model),
        "pymilo-ml-category": ml_category
    }


def deserialize_possible_ml_model(serialized_model):
    """
    Deserialize the given object if it is a previously serialized ML model.

    :param serialized_model: given obj to check
    :type serialized_model: obj
    :return: ML model flag and deserialized result
    """
    if check_str_in_iterable("pymilo-inner-model-type", serialized_model):
        _, transporter = get_transporter(serialized_model["pymilo-ml-category"])
        from ..transporters.transporter import Command
        return True, transporter({
            "data": serialized_model["pymilo-inner-model-data"],
            "type": serialized_model["pymilo-inner-model-type"]
        }, Command.DESERIALIZE, is_inner_model=True)
    else:
        return False, serialized_model
