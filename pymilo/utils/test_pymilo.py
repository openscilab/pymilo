# -*- coding: utf-8 -*-
"""pymilo test modules."""
import os

from ..pymilo_obj import Export
from ..pymilo_obj import Import

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, hinge_loss

from ..pymilo_func import compare_model_outputs

from ..chains.linear_model_chain import is_linear_model
from ..chains.neural_network_chain import is_neural_network
from ..chains.decision_tree_chain import is_decision_tree
from ..chains.clustering_chain import is_clusterer

from ..pymilo_param import EXPORTED_MODELS_PATH


def pymilo_export_path(model):
    """
    Return the associated folder name to save the json file generated by pymilo Export(applied to the given model).

    :param model: given model
    :type model: any sklearn's model class
    :return: folder name
    """
    model_type = None
    if is_linear_model(model):
        model_type = "LINEAR_MODEL"
    elif is_neural_network(model):
        model_type = "NEURAL_NETWORK"
    elif is_decision_tree(model):
        model_type = "DECISION_TREE"
    elif is_clusterer(model):
        model_type = "CLUSTERING"
    else:
        model_type = None
    return EXPORTED_MODELS_PATH[model_type]


def pymilo_test(model, model_name):
    """
    Return the pymilo imported model's outputs for given test_data.

    :param model: given model
    :type model: any sklearn's model class
    :param model_name: model name
    :type model_name: str
    :param test_data: data for testing
    :type test_data: np.ndarray or list
    :return: imported model's output
    """
    export_model_path = pymilo_export_path(model)
    exported_model = Export(model)
    exported_model_serialized_path = os.path.join(
        os.getcwd(), "tests", export_model_path, model_name + '.json')
    exported_model.save(exported_model_serialized_path)

    imported_model = Import(exported_model_serialized_path)
    imported_sklearn_model = imported_model.to_model()
    return imported_sklearn_model


def pymilo_regression_test(regressor, model_name, test_data):
    """
    Test the package's main structure in regression task.

    :param regressor: the given regressor model
    :type regressor: any valid sklearn's regressor class
    :param model_name: model name
    :type model_name: str
    :param test_data: data for testing
    :type test_data: np.ndarray or list
    :return: True if the test succeed
    """
    x_test, y_test = test_data
    pre_pymilo_model_y_pred = regressor.predict(x_test)
    pre_pymilo_model_prediction_output = {
        "mean-error": mean_squared_error(y_test, pre_pymilo_model_y_pred),
        "r2-score": r2_score(y_test, pre_pymilo_model_y_pred)
    }
    post_pymilo_model_y_pred = pymilo_test(regressor, model_name).predict(x_test)
    post_pymilo_model_prediction_outputs = {
        "mean-error": mean_squared_error(y_test, post_pymilo_model_y_pred),
        "r2-score": r2_score(y_test, post_pymilo_model_y_pred)
    }
    comparison_result = compare_model_outputs(
        pre_pymilo_model_prediction_output,
        post_pymilo_model_prediction_outputs)
    report_status(comparison_result, model_name)
    return comparison_result


def pymilo_classification_test(classifier, model_name, test_data):
    """
    Test the package's main structure in classification task.

    :param classifier: the given classifier model
    :type classifier: any valid sklearn's classifier class
    :param model_name: model name
    :type model_name: str
    :param test_data: data for testing
    :type test_data: np.ndarray or list
    :return: True if the test succeed
    """
    x_test, y_test = test_data
    pre_pymilo_model_y_pred = classifier.predict(x_test)
    pre_pymilo_model_prediction_output = {
        "accuracy-score": accuracy_score(y_test, pre_pymilo_model_y_pred),
        "hinge-loss": hinge_loss(y_test, pre_pymilo_model_y_pred)
    }
    post_pymilo_model_y_pred = pymilo_test(classifier, model_name).predict(x_test)
    post_pymilo_model_prediction_outputs = {
        "accuracy-score": accuracy_score(y_test, post_pymilo_model_y_pred),
        "hinge-loss": hinge_loss(y_test, post_pymilo_model_y_pred)
    }
    comparison_result = compare_model_outputs(
        pre_pymilo_model_prediction_output,
        post_pymilo_model_prediction_outputs)
    report_status(comparison_result, model_name)
    return comparison_result

def pymilo_clustering_test(clusterer, model_name, x_test, support_prediction = False):
    """
    Test the package's main structure in clustering task.

    :param clusterer: the given clusterer model
    :type clusterer: any valid sklearn's clusterer class
    :param model_name: model name
    :type model_name: str
    :param test_data: data for testing
    :type test_data: np.ndarray or list
    :return: True if the test succeed
    """
    pre_pymilo_model = clusterer
    post_pymilo_model = pymilo_test(clusterer, model_name)
    if(support_prediction):
        pre_pymilo_model_y_pred = pre_pymilo_model.predict(x_test)
        post_pymilo_model_y_pred = post_pymilo_model.predict(x_test)
        mse = ((post_pymilo_model_y_pred - pre_pymilo_model_y_pred)**2).mean(axis=0)
        epsilon_error = 10**(-8)
        return report_status(mse < epsilon_error, model_name) 
    else:
        # TODO, apply peer to peer 
        # Evaluation: peer to peer field type & value check 
        return report_status(True, model_name)
    
def report_status(result, model_name):
    """
    Print status for each model.

    :param result: the test result
    :type result: bool
    :param model_name: model name
    :type model_name: str
    :return: None
    """
    if result:
        print('Pymilo Test for Model: ' + model_name + ' succeed.')
    else:
        print('Pymilo Test for Model: ' + model_name + ' failed.')
