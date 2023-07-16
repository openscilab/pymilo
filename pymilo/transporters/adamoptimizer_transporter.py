# -*- coding: utf-8 -*-
"""PyMilo Adamoptimizer object transporter."""
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter

# Handling sklearn.neural_network._stochastic_optimizers.AdamOptimizer


class AdamOptimizerTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle AdamOptimizer field."""

    def serialize(self, data, key, model_type):
        """
        Serialize instances of the AdamOptimizer class.

        Record the `learning_rate`, `beta_1`, `beta_2` and `epsilon` fields of AdamOptimizer object.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which its data dictionary is given as the data param.
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if ((model_type == "MLPRegressor" and key ==
             "_optimizer" and isinstance(data[key], AdamOptimizer))):
            optimizer = data[key]
            data[key] = {
                'params': {
                    'learning_rate': optimizer.learning_rate,
                    'beta_1': optimizer.beta_1,
                    'beta_2': optimizer.beta_2,
                    'epsilon': optimizer.epsilon
                }
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize the special _optimizer field of the AdamOptimizer.

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
        :param model_type: the model type of the ML model, which internal serialized data dictionary is given as the data param
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        # if model_type != "MLPRegressor":
        #    return content
        if is_primitive(content) or isinstance(content, type(None)):
            return content
        if not check_str_in_iterable("_optimizer", content):
            return content

        optimizer = content['params']
        return AdamOptimizer(
            learning_rate=optimizer['learning_rate'],
            beta_1=optimizer['beta_1'],
            beta_2=optimizer['beta_2'],
            epsilon=optimizer['epsilon'])
