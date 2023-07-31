# -*- coding: utf-8 -*-
"""PyMilo SGDOptimizer object transporter."""
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter


class SGDOptimizerTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle SGDOptimizer field."""

    def serialize(self, data, key, model_type):
        """
        Serialize instances of the SGDOptimizer class.

        Record the `learning_rate`, `momentum`, `decay` and `nesterov` fields of random state object.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], SGDOptimizer):
            optimizer = data[key]
            data[key] = {
                'params': {
                    'type': "SGDOptimizer",
                    'learning_rate': optimizer.learning_rate,
                    'momentum': optimizer.momentum,
                    'decay': optimizer.decay,
                    'nesterov': optimizer.nesterov
                }
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

        if (key == "_optimizer" and (model_type ==
                                     "MLPRegressor" or model_type == "MLPClassifier")):
            optimizer = content['params']
            if (optimizer["type"] == "SGDOptimizer"):
                return SGDOptimizer(
                    learning_rate=optimizer['learning_rate'],
                    momentum=optimizer['momentum'],
                    decay=optimizer['decay'],
                    nesterov=optimizer['nesterov'])
            else:
                return content
        else:
            return content
