# -*- coding: utf-8 -*-
"""PyMilo Bunch transporter."""

from ..utils.util import check_str_in_iterable
from .transporter import AbstractTransporter
bunch_support = False
try:
    from sklearn.utils._bunch import Bunch
    bunch_support = True
except BaseException:
    pass


class BunchTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle Bunch objects."""

    def serialize(self, data, key, model_type):
        """
        Serialize Bunch object.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if bunch_support and isinstance(data[key], Bunch):
            bunch = data[key]
            dicted_bunch = {}
            dicted_bunch["pymiloed-data-structure"] = "Bunch"
            _dict = {}
            for key, value in bunch.items():
                _dict[key] = value
            dicted_bunch["pymiloed-data"] = _dict
            return dicted_bunch

        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized Bunch object.

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
        if bunch_support and check_str_in_iterable(
            "pymiloed-data-structure",
                content) and content["pymiloed-data-structure"] == "Bunch":
            bunch = Bunch()
            dicted_bunch = content["pymiloed-data"]
            for key, value in dicted_bunch.items():
                bunch[key] = value
            return bunch
        else:
            return content
