# -*- coding: utf-8 -*-
"""PyMilo RandomState(MT19937) object transporter."""
import numpy as np
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter

# Handling Numpy's RandomState


class RandomStateTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle RandomState field."""

    def serialize(self, data, key, model_type):
        """
        Serialize instances of the RandomState class.

        Record the `state` associated fields of random state object.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which its data dictionary is given as the data param.
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
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
        """
        Deserialize the previously serialized RandomState object.

        The associated _random_state field of the pymilo serialized NN model, is extracted through
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
