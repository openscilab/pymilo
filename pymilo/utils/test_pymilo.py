# -*- coding: utf-8 -*-
"""pymilo test modules."""
import os

from ..pymilo_obj import Export
from ..pymilo_obj import Import

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, hinge_loss

from ..pymilo_func import compare_model_outputs


def test_pymilo(model, model_name, test_data):
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
    x_test, _ = test_data

    exported_model = Export(model)
    exported_model_serialized_path = os.path.join(
        os.getcwd(), "tests", "exported_models", f'{model_name}.json')
    exported_model.save(exported_model_serialized_path)

    imported_model = Import(exported_model_serialized_path)
    imported_sklearn_model = imported_model.to_model()
    return imported_sklearn_model.predict(x_test)


def test_pymilo_regression(regressor, model_name, test_data):
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
    post_pymilo_model_y_pred = test_pymilo(regressor, model_name, test_data)
    post_pymilo_model_prediction_outputs = {
        "mean-error": mean_squared_error(y_test, post_pymilo_model_y_pred),
        "r2-score": r2_score(y_test, post_pymilo_model_y_pred)
    }
    comparison_result = compare_model_outputs(
        pre_pymilo_model_prediction_output,
        post_pymilo_model_prediction_outputs)
    report_status(comparison_result, model_name)
    return comparison_result


def test_pymilo_classification(classifier, model_name, test_data):
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
    post_pymilo_model_y_pred = test_pymilo(classifier, model_name, test_data)
    post_pymilo_model_prediction_outputs = {
        "accuracy-score": accuracy_score(y_test, post_pymilo_model_y_pred),
        "hinge-loss": hinge_loss(y_test, post_pymilo_model_y_pred)
    }
    comparison_result = compare_model_outputs(
        pre_pymilo_model_prediction_output,
        post_pymilo_model_prediction_outputs)
    report_status(comparison_result, model_name)
    return comparison_result


def report_status(result, model_name):
    """
    Print status for each model.

    :param result: the test result
    :type result: bool
    :param model_name: model name
    :type model_name: str
    :return: None
    """
    if (result):
        print(f'Pymilo Test for Model:{model_name} succeed.')
    else:
        print(f'Pymilo Test for Model:{model_name} failed.')
