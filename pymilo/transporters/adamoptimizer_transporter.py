# -*- coding: utf-8 -*-
"""PyMilo Adamoptimizer object transporter."""
from sklearn.neural_network._stochastic_optimizers import AdamOptimizer
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter

# Handling sklearn.neural_network._stochastic_optimizers.AdamOptimizer


class AdamOptimizerTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle AdamOptimizer field."""

    def serialize(self, data, key, model_type):

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
