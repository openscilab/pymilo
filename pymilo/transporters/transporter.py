from ..utils.util import get_sklearn_type
from abc import ABC, abstractmethod
from enum import Enum
from ..utils.util import is_primitive, check_str_in_iterable


class Command(Enum):
    """
    Command is an enum class used to determine the type of transportation.
    """
    SERIALIZE = 1
    DESERIALZIE = 2

# Transporter Interface


class Transporter(ABC):

    @abstractmethod

    def serialize(self, data, key, model_type):
        """
        Serialize the data[key] of the given model which it's type is model_type.
        basically in order to fully serialize a model, we should traverse over all the keys of it's data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dictionary
        :param key: the special key of the data param, which we're going to serialize it's value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which it's data dictionary is given as the data param.
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        pass

    @abstractmethod
    def deserialize(self, data, key, model_type):
        """
        deserialize the data[key] of the given model which it's type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of it's serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by 
        pymilo export.
        :type data: dictionary
        :param key: the special key of the data param, which we're going to deserialize it's value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which it's internal serialized data dictionary is given as the data param.
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        pass

    @abstractmethod
    def transport(self, request, command):
        """
        transport(either serializes or deserializes) the request according to the given command.
        basically in order to fully transport a request, we should traverse over all the keys of it's internal data dictionary and
        pass it through the chain of associated transporters to get fully transported.

        :param request: either a ML model object itself(when command is serialize) or an object associated with the json string of a pymilo serialized ML model(when command is deserialize)
        :type request: object
        :param command: determines the type of transportation, it can be either Serialize or Deserialize.
        :type command: Command class
        :return: pymilo transported output of data[key]
        """
        pass


class AbstractTransporter(Transporter):

    def bypass(self, content):
        """
        determine whether to bypass transporting on this content or not.
        :param content: either a ML model object's internal data dictionary or an object associated with the json string of a pymilo serialized ML model.
        :type content: object
        :return: boolean, whether to bypass or not
        """
        if is_primitive(content):
            return False

        if check_str_in_iterable("by-pass", content):
            return content["by-pass"]
        else:
            return False

    def transport(self, request, command, is_inner_model=False):
        """
        transport(either serializes or deserializes) the request according to the given command.
        basically in order to fully transport a request, we should traverse over all the keys of it's internal data dictionary and
        pass it through the chain of associated transporters to get fully transported.

        :param request: either a ML model object itself(when command is serialize) or an object associated with the json string of a pymilo serialized ML model(when command is deserialize)
        :type request: object
        :param command: determines the type of transportation, it can be either Serialize or Deserialize.
        :type command: Command class
        :return: pymilo transported output of data[key]
        """
        if command == Command.SERIALIZE:
            # request is a sklearn model
            data = request.__dict__
            for key in data.keys():
                if self.bypass(data[key]):
                    continue  # by-pass!!
                data[key] = self.serialize(
                    data, key, get_sklearn_type(request))

        elif command == Command.DESERIALZIE:
            # request is a pymilo-created import object
            data = None
            model_type = None
            if is_inner_model:
                data = request["data"]
                model_type = request["type"]
            else:
                data = request.data
                model_type = request.type
            for key in data.keys():
                data[key] = self.deserialize(data, key, model_type)
            return

        else:
            # TODO error handeling.
            return None
