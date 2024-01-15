# -*- coding: utf-8 -*-
"""utility module."""
from numpy import ndarray
from inspect import signature
import importlib


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
    Check if the specified string field exists in content, which is supposed to be an iterable object.

    :param field: given string field
    :type field: str
    :param content: given supposed to be an iterable object
    :type content: obj
    :return: True if associated field is an iterable string in content and False otherwise.
    """
    if not is_iterable(content):
        return False
    if isinstance(content, ndarray):
        # https://stackoverflow.com/questions/40659212/futurewarning-elementwise-comparison-failed-returning-scalar-but-in-the-futur.
        return False
    else:
        return field in content


def get_homogeneous_type(seq):
    """
    Check if the given sequence's inner items have the same type or not and if they do, return the associated type.

    :param seq: given sequence
    :type seq: sequence

    :return: Tuple of (True, inner_type) or (False, None)
    """
    iseq = iter(seq)
    first_type = type(next(iseq))
    return (
        True,
        first_type) if all(
        (type(x) is first_type) for x in iseq) else (
            False,
        None)


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
