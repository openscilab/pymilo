# -*- coding: utf-8 -*-
"""PyMilo RandomState(MT19937) object transporter."""
import numpy as np
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter

# Handling Numpy's RandomState

class RandomStateTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle RandomState field."""

    def serialize(self, data, key, model_type):

        if (
            (model_type == "MLPRegressor" and key == "_random_state") 
        ):
            rng = data[key]
            data[key] = {
                'state': (
                rng.get_state()[0],
                rng.get_state()[1].tolist(),
                rng.get_state()[2],
                rng.get_state()[3],
                rng.get_state()[4]
                )
            }
        return data[key]

    def deserialize(self, data, key, model_type):

        content = data[key]
        # if model_type != "MLPRegressor":
        #    return content 
        if is_primitive(content) or isinstance(content, type(None)):
            return content
        if not check_str_in_iterable("_random_state", content):
            return content
    
        rng_state = content['state']
        rng_state = (
            rng_state[0], 
            np.array(rng_state[1]),
            rng_state[2],
            rng_state[3],
            rng_state[4]
            )
        _rng = np.random.RandomState()
        _rng.set_state(rng_state)
        return _rng 
