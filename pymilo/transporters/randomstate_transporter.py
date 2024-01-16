# -*- coding: utf-8 -*-
"""PyMilo RandomState(MT19937) object transporter."""
import numpy as np
from .transporter import AbstractTransporter


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
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], np.random.RandomState):
            inner_random_state = data[key]
            data[key] = {
                'state': (
                    inner_random_state.get_state()[0],
                    inner_random_state.get_state()[1].tolist(),
                    inner_random_state.get_state()[2],
                    inner_random_state.get_state()[3],
                    inner_random_state.get_state()[4]
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
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]

        if key == "_random_state" and (
                model_type == "MLPRegressor" or model_type == "MLPClassifier"):
            inner_random_state = content['state']
            inner_random_state = (
                inner_random_state[0],
                np.array(inner_random_state[1]),
                inner_random_state[2],
                inner_random_state[3],
                inner_random_state[4]
            )
            _random_state = np.random.RandomState()
            _random_state.set_state(inner_random_state)
            return _random_state
        else:
            return content
