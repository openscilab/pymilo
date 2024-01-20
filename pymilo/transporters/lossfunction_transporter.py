# -*- coding: utf-8 -*-
"""PyMilo Loss function transporter."""
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter


class LossFunctionTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle Loss function field."""

    def serialize(self, data, key, model_type):
        """
        Serialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.

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
        if (
            (model_type == "SGDClassifier" and (key == "loss_function_" or key == "_loss_function_")) or
            (model_type == "SGDOneClassSVM" and (key == "loss_function_" or key == "_loss_function_")) or
            (model_type == "Perceptron" and (key == "loss_function_" or key == "_loss_function_")) or
            (model_type == "PassiveAggressiveClassifier" and (key == "loss_function_" or key == "_loss_function_"))
        ):
            data[key] = {
                "loss": data["loss"]
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.

        the associated loss_function_ field of the pymilo serialized model, is extracted through
        the SGDClassifier's _get_loss_function function with enough feeding of the needed inputs.

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
        if is_primitive(content) or isinstance(content, type(None)):
            return content
        if not check_str_in_iterable("loss", content):
            return content
        return SGDClassifier(
            loss=content["loss"])._get_loss_function(
            content["loss"])
