# -*- coding: utf-8 -*-
"""PyMilo Transporter."""
from ..utils.util import get_sklearn_type
from abc import ABC, abstractmethod
from enum import Enum
from ..utils.util import is_primitive, check_str_in_iterable


class Command(Enum):
    """Command is an enum class used to determine the type of transportation."""

    SERIALIZE = 1
    DESERIALIZE = 2


class Transporter(ABC):
    """
    Transporter Interface.

    Each Transporter transports(either serializes or deserializes) the input according to the given command.
    """

    @abstractmethod
    def serialize(self, data, key, model_type):
        """
        Serialize the data[key] of the given model which type is model_type.

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

    @abstractmethod
    def deserialize(self, data, key, model_type):
        """
        Deserialize the data[key] of the given model which type is model_type.

        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model
            which is generated previously by pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which internal serialized data dictionary is given as the data param
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """

    @abstractmethod
    def bypass(self, content):
        """
        Determine whether to bypass transporting on this content or not.

        :param content: either a ML model object's internal data dictionary(.__dict__) or an object associated with the json string of a pymilo serialized ML model.
        :type content: object
        :return: boolean, whether to bypass or not
        """

    @abstractmethod
    def reset(self):
        """
        Reset internal data structures of the transport object.

        Some Transporters may be stateful and have internal data structures getting filled during transportation.

        :return: None
        """

    @abstractmethod
    def transport(self, request, command):
        """
        Either serializes or deserializes the request according to the given command.

        basically in order to fully transport a request, we should traverse over all the keys of its internal data dictionary and
        pass it through the chain of associated transporters to get fully transported.

        :param request: either a ML model object itself(when command is serialize) or
        an object associated with the json string of a pymilo serialized ML model(when command is deserialize)
        :type request: object
        :param command: determines the type of transportation, it can be either Serialize or Deserialize
        :type command: Command class
        :return: pymilo transported output of data[key]
        """


class AbstractTransporter(Transporter):
    """Abstract Transporter with the implementation of the traversing through the given input according to the associated command."""

    def bypass(self, content):
        """
        Determine whether to bypass transporting on this content or not.

        :param content: either a ML model object's internal data dictionary or an object associated with the json string of a pymilo serialized ML model.
        :type content: object
        :return: boolean, whether to bypass or not
        """
        if is_primitive(content):
            return False

        if check_str_in_iterable("pymilo-bypass", content):
            return content["pymilo-bypass"]
        else:
            return False

    def transport(self, request, command, is_inner_model=False):
        """
        Either serializes or deserializes the request according to the given command.

        basically in order to fully transport a request, we should traverse over all the keys of its internal data dictionary and
        pass it through the chain of associated transporters to get fully transported.

        :param request: either a ML model object itself(when command is serialize) or an object associated with the json string of a pymilo serialized ML model(when command is deserialize)
        :type request: object
        :param command: determines the type of transportation, it can be either Serialize or Deserialize
        :type command: Command class
        :param is_inner_model: determines whether it is an inner linear model of a super ml model
        :type is_inner_model: boolean
        :return: pymilo transported output of data[key]
        """
        if command == Command.SERIALIZE:
            # request is a sklearn model
            data = request.__dict__
            for key in data:
                if self.bypass(data[key]):
                    continue  # by-pass!!
                data[key] = self.serialize(
                    data, key, get_sklearn_type(request))
            self.reset()

        elif command == Command.DESERIALIZE:
            # request is a pymilo-created import object
            data = None
            model_type = None
            if is_inner_model:
                data = request["data"]
                model_type = request["type"]
            else:
                data = request.data
                model_type = request.type
            for key in data:
                data[key] = self.deserialize(data, key, model_type)
            self.reset()
            return

        else:
            # TODO error handling.
            return None

    def reset(self):
        """
        Reset internal data structures of the transport object.

        Some Transporters may be stateful and have internal data structures getting filled during transportation.

        :return: None
        """
        return
