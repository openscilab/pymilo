# -*- coding: utf-8 -*-
"""PyMiloServer for RESTFull protocol."""
from ..pymilo_obj import Export, Import
from .encryptor import DummyEncryptor
from .compressor import DummyCompressor
from .communicator import RESTServerCommunicator
from .param import PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class PymiloServer:
    """Facilitate streaming the ML models."""

    def __init__(self, port=8000):
        """
        Initialize the Pymilo PymiloServer instance.

        :param port: the port to which PyMiloServer listens
        :type port: int
        :return: an instance of the PymiloServer class
        """
        self._model = None
        self._compressor = DummyCompressor()
        self._encryptor = DummyEncryptor()
        self._communicator = RESTServerCommunicator(ps=self, port=port)

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
        attribute = request.attribute
        retrieved_attribute = getattr(self._model, attribute, None)
        if retrieved_attribute is None:
            raise Exception(PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE)
        arguments = {
            'args': request.args,
            'kwargs': request.kwargs
        }
        args = gdst.deserialize(arguments, 'args', None)
        kwargs = gdst.deserialize(arguments, 'kwargs', None)
        output = retrieved_attribute(*args, **kwargs)
        if isinstance(output, type(self._model)):
            self._model = output
            return None
        return gdst.serialize({'output': output}, 'output', None)
