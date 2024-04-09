# -*- coding: utf-8 -*-
"""PyMilo Generator transporter."""
from numpy.random._generator import Generator
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter
from numpy.random import default_rng


class GeneratorTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle Generator objects."""

    def serialize(self, data, key, model_type):
        """
        Serialize Generator object.

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
        if isinstance(data[key], Generator):
            generator = data[key]
            data[key] = {
                "pymilo-bypass": True,
                "pymilo-generator": {
                    "state": generator.__getstate__()
                }
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized Generator object.

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

        if check_str_in_iterable("pymilo-generator", content):
            serialized_generator = content["pymilo-generator"]
            generator = default_rng(0)
            generator.__setstate__(serialized_generator["state"])
            return generator

        return content
