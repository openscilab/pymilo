# -*- coding: utf-8 -*-
"""utility module."""
import requests
import importlib
from inspect import signature
from ..pymilo_param import DOWNLOAD_MODEL_FAILED, INVALID_DOWNLOADED_MODEL, SKLEARN_SUPPORTED_CATEGORIES


def get_sklearn_type(model):
    """
    Return sklearn model type.

    :param model: sklearn model
    :type model: any sklearn's model class
    :return: model type as str
    """
    raw_type = type(model)
    return str(raw_type).split(".")[-1][:-2]


def is_primitive(obj):
    """
    Check if the given object is primitive.

    :param obj: given object
    :type obj: any valid type
    :return: True if object is primitive
    """
    if isinstance(obj, dict):
        return False
    return not hasattr(obj, '__dict__')


def is_iterable(obj):
    """
    Check if the given object is iterable.

    :param obj: given object
    :type obj: any valid type
    :return: True if object is iterable
    """
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def check_str_in_iterable(field, content):
    """
    Check if the specified string field exists in content, which is supposed to be a dictionary.

    :param field: given string field
    :type field: str
    :param content: given supposed to be a dictionary
    :type content: obj
    :return: True if associated field is an iterable string in content and False otherwise.
    """
    if isinstance(content, dict):
        return field in content
    else:
        return False


def get_homogeneous_type(seq):
    """
    Check if the given sequence's inner items have the same type or not and if they do, return the associated type.

    :param seq: given sequence
    :type seq: sequence

    :return: Tuple of (True, inner_type) or (False, None)
    """
    iseq = iter(seq)
    first_type = type(next(iseq))
    if all((isinstance(x, first_type)) for x in iseq):
        return True, first_type
    else:
        return False, None


def all_same(arr):
    """
    Check if the given array's items are the same or not.

    :param arr: given array
    :type arr: array

    :return: bool
    """
    return all(x == arr[0] for x in arr)


def import_function(module_name, function_name):
    """
    Import function with name function_name from module called module_name.

    :param module_name: module to import the function from
    :type module_name: str
    :param function_name: function's name to get imported
    :type function_name: str

    :return: function
    """
    module = importlib.import_module(module_name)
    function = getattr(module, function_name)
    return function


def has_named_parameter(func, param_name):
    """
    Check whether the given function has a parameter named param_name or not.

    :param func: function to check it's params
    :type func: function
    :param param_name: parameter's name
    :type param_name: str

    :return: boolean
    """
    _signature = signature(func)
    parameter_names = [p.name for p in _signature.parameters.values()]
    return param_name in parameter_names


def prefix_list(list1, list2):
    """
    Check whether the list2 list is list1 sublist of the a list.

    :param list1: outer list
    :type list1: list
    :param list2: inner list
    :type list2: list

    :return: boolean
    """
    if len(list1) < len(list2):
        return False
    return all(list1[j] == list2[j] for j in range(len(list2)))


def download_model(url):
    """
    Download the model from the given url.

    :param url: url to exported JSON file
    :type url: str

    :return: obj
    """
    s = requests.Session()
    retries = requests.adapters.Retry(
        total=5,
        backoff_factor=0.1,
        status_forcelist=[500, 502, 503, 504]
    )
    s.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
    s.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))
    try:
        response = s.get(url)
    except Exception:
        raise Exception(DOWNLOAD_MODEL_FAILED)
    try:
        if response.status_code == 200:
            return response.json()
    except ValueError:
        raise Exception(INVALID_DOWNLOADED_MODEL)


def get_sklearn_class(model_name):
    """
    Return the sklearn class of the requested model name.

    :param model_name: model name
    :type model_name: str

    :return: sklearn ML model class
    """
    for _, category_models in SKLEARN_SUPPORTED_CATEGORIES.items():
        if model_name in category_models:
            return category_models[model_name]
    return None
