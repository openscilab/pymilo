# -*- coding: utf-8 -*-
"""PyMilo Function transporter."""

from ..utils.util import import_function, check_str_in_iterable
from .transporter import AbstractTransporter
from types import FunctionType, BuiltinFunctionType
from numpy import ufunc

array_function_dispatcher_support = False
try:
    from numpy.core._multiarray_umath import _ArrayFunctionDispatcher
    array_function_dispatcher_support = True
except BaseException:
    pass


class FunctionTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle function field transportation."""

    def serialize(self, data, key, model_type):
        """
        Serialize Function type fields.

        Record associated function's name and it's parent module in order to retrieve it accordingly.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], ufunc):
            function = data[key]
            data[key] = {
                "function_name": function.__name__,
                "function_module": "numpy",
            }
            return data[key]

        elif isinstance(data[key], (FunctionType, BuiltinFunctionType)) or (
                array_function_dispatcher_support and
                isinstance(data[key], _ArrayFunctionDispatcher)):
            function = data[key]
            data[key] = {
                "function_name": function.__name__,
                "function_module": function.__module__,
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize serialized function objects back to it's original function type object.

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by
        pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if check_str_in_iterable("function_name", content):
            return import_function(
                content["function_module"],
                content["function_name"]
            )
        else:
            return content
