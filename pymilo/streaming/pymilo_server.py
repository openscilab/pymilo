# -*- coding: utf-8 -*-
"""PyMiloServer for RESTFull protocol."""
from ..pymilo_obj import Export, Import
from .compressor import Compression
from .encryptor import DummyEncryptor
from .communicator import ServerCommunicationProtocol
from .param import PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class PymiloServer:
    """Facilitate streaming the ML models."""

    def __init__(
            self,
            model=None,
            port=8000,
            compressor=Compression.NULL,
            server_communicator=ServerCommunicationProtocol.REST,
    ):
        """
        Initialize the Pymilo PymiloServer instance.

        :param model: the ML model which will be streamed
        :type model: any
        :param port: the port to which PyMiloServer listens
        :type port: int
        :param compressor: the compression method to be used in client-server communications
        :type compressor: pymilo.streaming.compressor.Compression
        :return: an instance of the PymiloServer class
        """
        self._model = model
        self._compressor = compressor.value
        self._encryptor = DummyEncryptor()
        self.communicator = server_communicator.value(ps=self, port=port)

    def export_model(self):
        """
        Export the ML model to string json dump using PyMilo Export class.

        :return: str
        """
        return Export(self._model).to_json()

    def update_model(self, serialized_model):
        """
        Update the PyMilo Server's ML model.

        :param serialized_model: the json dump of a pymilo export ml model
        :type serialized_model: str
        :return: None
        """
        self._model = Import(file_adr=None, json_dump=serialized_model).to_model()

    def execute_model(self, request):
        """
        Execute the request attribute call from PyMilo Client.

        :param request: request obj containing requested attribute to call with the associated args and kwargs
        :type request: obj
        :return: str | dict
        """
        gdst = GeneralDataStructureTransporter()
        attribute = request["attribute"] if isinstance(request, dict) else request.attribute
        retrieved_attribute = getattr(self._model, attribute, None)
        if retrieved_attribute is None:
            raise Exception(PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE)
        arguments = {
            'args': request["args"] if isinstance(request, dict) else request.args,
            'kwargs': request["kwargs"] if isinstance(request, dict) else request.kwargs,
        }
        args = gdst.deserialize(arguments, 'args', None)
        kwargs = gdst.deserialize(arguments, 'kwargs', None)
        output = retrieved_attribute(*args, **kwargs)
        if isinstance(output, type(self._model)):
            self._model = output
            return None
        return gdst.serialize({'output': output}, 'output', None)

    def is_callable_attribute(self, request):
        """
        Check whether the requested attribute is callable or not.

        :param request: request obj containing requested attribute to check it's type
        :type request: obj
        :return: True if it is callable False otherwise
        """
        attribute = request["attribute"] if isinstance(request, dict) else request.attribute
        retrieved_attribute = getattr(self._model, attribute, None)
        if callable(retrieved_attribute):
            return True, None
        else:
            return False, GeneralDataStructureTransporter().serialize({'output': retrieved_attribute}, 'output', None)
