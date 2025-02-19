# -*- coding: utf-8 -*-
"""PyMiloServer for RESTFull protocol."""
from ..pymilo_obj import Export, Import
from .compressor import Compression
from .encryptor import DummyEncryptor
from .communicator import CommunicationProtocol
from .param import PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class PymiloServer:
    """Facilitate streaming the ML models."""

    def __init__(
            self,
            port=8000,
            compressor=Compression.NULL,
            communication_protocol=CommunicationProtocol.REST,
    ):
        """
        Initialize the Pymilo PymiloServer instance.

        :param model: the ML model which will be streamed
        :type model: any
        :param port: the port to which PyMiloServer listens
        :type port: int
        :param compressor: the compression method to be used in client-server communications
        :type compressor: pymilo.streaming.compressor.Compression
        :param communication_protocol: The communication protocol to be used by PymiloServer
        :type communication_protocol: pymilo.streaming.communicator.CommunicationProtocol
        :return: an instance of the PymiloServer class
        """
        self._compressor = compressor.value
        self._encryptor = DummyEncryptor()
        self.communicator = communication_protocol.value["SERVER"](ps=self, port=port)
        # In-memory storage (replace with a database for persistence)
        self._clients = {}

    def export_model(self, client_id, ml_model_id):
        """
        Export the ML model to string json dump using PyMilo Export class.

        :return: str
        """
        return Export(self._clients[client_id][ml_model_id]).to_json()

    def update_model(self, client_id, ml_model_id, serialized_model):
        """
        Update the PyMilo Server's ML model.

        :param serialized_model: the json dump of a pymilo export ml model
        :type serialized_model: str
        :return: None
        """
        self._clients[client_id][ml_model_id] = Import(file_adr=None, json_dump=serialized_model).to_model()

    def execute_model(self, request):
        """
        Execute the request attribute call from PyMilo Client.

        :param request: request obj containing requested attribute to call with the associated args and kwargs
        :type request: obj
        :return: str | dict
        """
        gdst = GeneralDataStructureTransporter()
        attribute = request["attribute"] if isinstance(request, dict) else request.attribute
        _client_id = request["client_id"] if isinstance(request, dict) else request.client_id
        _ml_model_id = request["ml_model_id"] if isinstance(request, dict) else request.ml_model_id
        _ml_model = self._clients[_client_id][_ml_model_id]
        retrieved_attribute = getattr(_ml_model, attribute, None)
        if retrieved_attribute is None:
            raise Exception(PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE)
        arguments = {
            'args': request["args"] if isinstance(request, dict) else request.args,
            'kwargs': request["kwargs"] if isinstance(request, dict) else request.kwargs,
        }
        args = gdst.deserialize(arguments, 'args', None)
        kwargs = gdst.deserialize(arguments, 'kwargs', None)
        output = retrieved_attribute(*args, **kwargs)
        if isinstance(output, type(_ml_model)):
            self._clients[_client_id][_ml_model_id] = output
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
        _client_id = request["client_id"] if isinstance(request, dict) else request.client_id
        _ml_model_id = request["ml_model_id"] if isinstance(request, dict) else request.ml_model_id
        _ml_model = self._clients[_client_id][_ml_model_id]
        retrieved_attribute = getattr(_ml_model, attribute, None)
        if callable(retrieved_attribute):
            return True, None
        else:
            return False, GeneralDataStructureTransporter().serialize({'output': retrieved_attribute}, 'output', None)

    def _validate_id(self, client_id, ml_model_id):
        if client_id not in self._clients:
            return False, "The given client_id is invalid."
        if ml_model_id not in self._clients[client_id]:
            return False, "The given client_id is valid but requested ml_model_id is invalid."
        return True, None

    def init_client(self, client_id):
        if client_id in self._clients:
            return False, f"The client with client_id: {client_id} already exists."
        self._clients[client_id] = {}
        return True, None

    def remove_client(self, client_id):
        if client_id not in self._clients:
            return False, f"The client with client_id: {client_id} doesn't exist."
        del self._clients[client_id]
        return True, None

    def get_clients(self):
        return [id for id in self._clients.keys()]

    def init_ml_model(self, client_id, ml_model_id):
        if client_id not in self._clients:
            return False, "The given client_id is invalid."
        
        if ml_model_id in self._clients[client_id]:
            return False, f"The given ml_model_id: {ml_model_id} already exists within ml models of the client with client_id of {client_id}."
        
        self._clients[client_id][ml_model_id] = {}
        return True, None

    def set_ml_model(self, client_id, ml_model_id, ml_model):
        self._clients[client_id][ml_model_id] = ml_model

    def remove_ml_model(self, client_id, ml_model_id):
        if client_id not in self._clients:
            return False, "The given client_id is invalid."
        
        if ml_model_id not in self._clients[client_id]:
            return False, f"The client with client_id: {client_id} doesn't have any model with ml_model_id of {ml_model_id}."
        
        del self._clients[client_id][ml_model_id]
        return True, None

    def get_ml_models(self, client_id):
        return [id for id in self._clients[client_id].keys()]