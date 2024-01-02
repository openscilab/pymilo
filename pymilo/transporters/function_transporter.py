# -*- coding: utf-8 -*-
"""PyMilo Function transporter."""

from .transporter import AbstractTransporter
from ..utils.util import import_function, is_iterable
import types
import collections

class FunctionTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle function field serialization."""

    def serialize(self, data, key, model_type):
        """
        Serialize Function type fields.

        Record the signature of the associated function in order to retrieve it accordingly.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], types.FunctionType):
            function = data[key]
            data[key] = {
                "function_name": function.__name__,
                "function_module": function.__module__,
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize the special _optimizer field of the SGDOptimizer.

        The associated _optimizer field of the pymilo serialized model, is extracted through
        it's previously serialized parameters.

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
        if isinstance(content, collections.abc.Iterable) and "function_name" in content:
            return import_function(
                content["function_module"],
                content["function_name"]
            )
        else:
            return content
