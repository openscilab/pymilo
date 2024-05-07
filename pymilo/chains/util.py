# -*- coding: utf-8 -*-
"""useful utilities for chains."""
from .linear_model_chain import transport_linear_model, is_linear_model
from .neural_network_chain import transport_neural_network, is_neural_network
from .decision_tree_chain import transport_decision_tree, is_decision_tree
from .clustering_chain import transport_clusterer, is_clusterer
from .naive_bayes_chain import transport_naive_bayes, is_naive_bayes
from .svm_chain import transport_svm, is_svm
from .neighbours_chain import transport_neighbor, is_neighbors


MODEL_TYPE_TRANSPORTER = {
    "LINEAR_MODEL": transport_linear_model,
    "NEURAL_NETWORK": transport_neural_network,
    "DECISION_TREE": transport_decision_tree,
    "CLUSTERING": transport_clusterer,
    "NAIVE_BAYES": transport_naive_bayes,
    "SVM": transport_svm,
    "NEIGHBORS": transport_neighbor
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

    if is_linear_model(model):
        return "LINEAR_MODEL", transport_linear_model
    elif is_neural_network(model):
        return "NEURAL_NETWORK", transport_neural_network
    elif is_decision_tree(model):
        return "DECISION_TREE", transport_decision_tree
    elif is_clusterer(model):
        return "CLUSTERING", transport_clusterer
    elif is_naive_bayes(model):
        return "NAIVE_BAYES", transport_naive_bayes
    elif is_svm(model):
        return "SVM", transport_svm
    elif is_neighbors(model):
        return "NEIGHBORS", transport_neighbor
    else:
        return None, None
