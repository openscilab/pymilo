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
    :return: tuple(ML_MODEL_CATEGORY, transporter function)
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
