# -*- coding: utf-8 -*-
"""Functions."""
import numpy as np
import sklearn

from .chains.util import get_transporter
from .transporters.transporter import Command
from .pymilo_param import SKLEARN_SUPPORTED_CATEGORIES, XGBOOST_SUPPORTED_CATEGORIES, NOT_SUPPORTED, OVERVIEW

_xgb = None
_xgboost_available = False
try:
    import xgboost as _xgb_imported
    _xgb = _xgb_imported
    _xgboost_available = True
except BaseException:
    _xgb = None
    _xgboost_available = False


def get_sklearn_version():
    """
    Return sklearn version.

    :return: sklearn version as a str
    """
    return sklearn.__version__


def get_xgboost_version():
    """
    Return xgboost version when available.

    :return: xgboost version as a str, or None when xgboost is not installed
    """
    if not _xgboost_available:
        return None
    return _xgb.__version__


def get_ml_model_data(model):
    """
    Return the serialized data of the given ML model.

    :param model: given model
    :type model: any supported ML model class
    :return: serialized model data
    """
    _, transporter = get_transporter(model)
    return transporter(model, Command.SERIALIZE)


def to_ml_model(import_obj):
    """
    Deserialize the imported object as an ML model.

    :param import_obj: given object
    :type import_obj: pymilo.Import
    :return: ML model
    """
    _, transporter = get_transporter(import_obj.type)
    return transporter(import_obj, Command.DESERIALIZE)


def get_framework_versions(model):
    """
    Return top-level framework version fields for the given model.

    Pure sklearn models emit `sklearn_version`. Pure native xgboost.Booster models emit
    `xgboost_version`. XGBoost sklearn-API wrappers emit both, since they sit on both stacks.

    :param model: given model
    :type model: any supported ML model
    :return: dict of framework version fields
    """
    category, _ = get_transporter(model)
    versions = {}
    if category == "XGBOOST":
        xgb_version = get_xgboost_version()
        if xgb_version is not None:
            versions["xgboost_version"] = xgb_version
        if not (_xgboost_available and isinstance(model, _xgb.Booster)):
            versions["sklearn_version"] = get_sklearn_version()
    else:
        versions["sklearn_version"] = get_sklearn_version()
    return versions


def compare_model_outputs(exported_output,
                          imported_output,
                          epsilon_error=10**(-8)):
    """
    Check if the given models outputs are the same.

    :param exported_output: exported model output
    :type exported_output: dict
    :param imported_output: imported model output
    :type imported_output: dict
    :param epsilon_error: error threshold for numeric comparisons
    :type epsilon_error: float
    :return: check result as bool
    """
    if len(exported_output) != len(imported_output):
        return False  # TODO: throw exception
    total_error = 0
    for key in exported_output:
        if key not in imported_output:
            return False  # TODO: throw exception
        total_error += np.abs(imported_output[key] - exported_output[key])
    return bool(np.abs(total_error) < epsilon_error)


def print_supported_ml_models():
    """
    Print the supported ML models categorized by type.

    :return: None
    """
    print("Supported Machine Learning Models:")
    all_categories = {}
    all_categories.update(SKLEARN_SUPPORTED_CATEGORIES)
    all_categories.update(XGBOOST_SUPPORTED_CATEGORIES)
    for category, table in all_categories.items():
        print(f"**{category}**:")
        for model_name in table:
            if table[model_name] != NOT_SUPPORTED:
                print(f"- {model_name}")


def pymilo_help():
    """
    Print PyMilo details.

    :return: None
    """
    print(OVERVIEW)
    print("Repo : https://github.com/openscilab/pymilo")
    print("Webpage : https://openscilab.com/\n")
