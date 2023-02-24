from ..utils.util import get_sklearn_type
from abc import ABC, abstractmethod
from enum import Enum
from ..utils.util import is_primitive, check_str_in_iterable

class Command(Enum):
    SERIALIZE = 1
    DESERIALZIE = 2

# Transporter Interface


class Transporter(ABC):

    @abstractmethod
    def serialize(self, data, key, model_type):
        pass

    @abstractmethod
    def deserialize(self, data, key, model_type):
        pass

    @abstractmethod
    def transport(self, request, command):
        pass


class AbstractTransporter(Transporter):

    def bypass(self, content):
        if is_primitive(content):
            return False

        if check_str_in_iterable("by-pass", content):
            return content["by-pass"]
        else:
            return False

    def transport(self, request, command, is_inner_model=False):
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
