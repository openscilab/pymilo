# -*- coding: utf-8 -*-
"""PyMilo SGDOptimizer object transporter."""
from sklearn.neural_network._stochastic_optimizers import SGDOptimizer
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter

# Handling sklearn.neural_network._stochastic_optimizers.SGDOptimizer

class SGDOptimizerTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle SGDOptimizer field."""

    def serialize(self, data, key, model_type):

        if (
            (model_type == "MLPRegressor" and key == "_optimizer" and isinstance(data[key], SGDOptimizer)) 
        ):
            optimizer = data[key]
            data[key] = {
                'params': {
                    'learning_rate': optimizer.learning_rate,
                    'momentum': optimizer.momentum,
                    'decay': optimizer.decay,
                    'nesterov': optimizer.nesterov
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
        return SGDOptimizer(learning_rate=optimizer['learning_rate'], momentum=optimizer['momentum'],
                                 decay=optimizer['decay'], nesterov=optimizer['nesterov'])
