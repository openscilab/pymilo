# -*- coding: utf-8 -*-
"""PyMilo OneHotEncoder transporter."""
from sklearn.preprocessing import OneHotEncoder
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter


class OneHotEncoderTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle OneHotEncoder objects."""

    def serialize(self, data, key, model_type):
        """
        Serialize OneHotEncoder object.

        serialize the data[key] of the given model which type is model_type.
        basically in order to fully serialize a model, we should traverse over all the keys of its data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which data dictionary is given as the data param
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], OneHotEncoder):
            data[key] = {
                "pymilo-onehotencoder": {
                    "sparse_output": data["sparse_output"]
                }
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized OneHotEncoder object.

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by
        pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which internal serialized data dictionary is given as the data param
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if is_primitive(content) or content is None:
            return content

        if check_str_in_iterable("pymilo-onehotencoder", content):
            return OneHotEncoder(sparse_output=content["pymilo-onehotencoder"]["sparse_output"])
        return content
