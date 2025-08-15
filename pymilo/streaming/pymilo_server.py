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
            host="127.0.0.1",
            compressor=Compression.NULL,
            communication_protocol=CommunicationProtocol.REST,
    ):
        """
        Initialize the Pymilo PymiloServer instance.

        :param model: the ML model which will be streamed
        :type model: any
        :param port: the port to which PyMiloServer listens
        :type port: int
        :param host: the url to which PyMilo Server listens
        :type host: str
        :param compressor: the compression method to be used in client-server communications
        :type compressor: pymilo.streaming.compressor.Compression
        :param communication_protocol: The communication protocol to be used by PymiloServer
        :type communication_protocol: pymilo.streaming.communicator.CommunicationProtocol
        :return: an instance of the PymiloServer class
        """
        self._compressor = compressor.value
        self._encryptor = DummyEncryptor()
        # In-memory storage (replace with a database for persistence)
        self.communicator = communication_protocol.value["SERVER"](ps=self, host=host, port=port)
        self._clients = {}
        self._allowance = {}

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
        """
        Validate the provided client ID and machine learning model ID.

        :param client_id: The ID of the client to validate.
        :type client_id: str
        :param ml_model_id: The ID of the machine learning model to validate.
        :type ml_model_id: str
        :return: A tuple containing a boolean indicating validity and an error message if invalid.
        """
        if client_id not in self._clients:
            return False, "The given client_id is invalid."
        if ml_model_id not in self._clients[client_id]:
            return False, "The given client_id is valid but requested ml_model_id is invalid."
        return True, None

    def init_client(self, client_id):
        """
        Initialize a new client with the given client ID.

        :param client_id: The ID of the client to initialize.
        :type client_id: str
        :return: A tuple containing a boolean indicating success and an error message if the client already exists.
        """
        if client_id in self._clients:
            return False, f"The client with client_id: {client_id} already exists."
        self._clients[client_id] = {}
        self._allowance[client_id] = {}
        return True, None

    def remove_client(self, client_id):
        """
        Remove an existing client by the given client ID.

        :param client_id: The ID of the client to remove.
        :type client_id: str
        :return: A tuple containing a boolean indicating success and an error message if the client does not exist.
        """
        if client_id not in self._clients:
            return False, f"The client with client_id: {client_id} doesn't exist."
        del self._clients[client_id]
        del self._allowance[client_id]
        return True, None

    def grant_access(self, allower_id, allowee_id, allowed_model_id):
        """
        Allow a client to access a specific machine learning model of another client.

        :param allower_id: The ID of the client granting access.
        :type allower_id: str
        :param allowee_id: The ID of the client being granted access.
        :type allowee_id: str
        :param allowed_model_id: The ID of the machine learning model to be accessed.
        :type allowed_model_id: str
        :return: A tuple containing a boolean indicating success and an error message if the operation fails.
        """
        if allower_id not in self._clients:
            return False, f"The allower client with client_id: {allower_id} doesn't exist."
        if allowee_id not in self._clients:
            return False, f"The allowee client with client_id: {allowee_id} doesn't exist."
        if allowed_model_id not in self._clients[allower_id]:
            return False, f"The model with ml_model_id: {allowed_model_id} doesn't exist for the allower client with client_id: {allower_id}."

        if allowed_model_id in self._allowance.get(allower_id).get(allowee_id, []):
            return False, f"The model with ml_model_id: {allowed_model_id} is already allowed for the allowee client with client_id: {allowee_id} by the allower client with client_id: {allower_id}."

        if allowee_id not in self._allowance[allower_id]:
            self._allowance[allower_id][allowee_id] = [allowed_model_id]
            return True, None

        self._allowance[allower_id][allowee_id].append(allowed_model_id)
        return True, None

    def revoke_access(self, allower_id, allowee_id, allowed_model_id=None):
        """
        Revoke a client's access to a specific machine learning model of another client.

        :param allower_id: The ID of the client revoking access.
        :type allower_id: str
        :param allowee_id: The ID of the client whose access is being revoked.
        :type allowee_id: str
        :param allowed_model_id: The ID of the machine learning model whose access is being revoked.
        :type allowed_model_id: str
        :return: A tuple containing a boolean indicating success and an error message if the operation fails.
        """
        if allower_id not in self._clients:
            return False, f"The allower client with client_id: {allower_id} doesn't exist."
        if allowee_id not in self._clients:
            return False, f"The allowee client with client_id: {allowee_id} doesn't exist."

        if allowed_model_id is None:
            if allowee_id in self._allowance[allower_id]:
                del self._allowance[allower_id][allowee_id]
            return True, None

        if allowed_model_id not in self._clients[allower_id]:
            return False, f"The model with ml_model_id: {allowed_model_id} doesn't exist for the allower client with client_id: {allower_id}."

        if allowee_id not in self._allowance[allower_id]:
            return False, f"The allowee client with client_id: {allowee_id} doesn't have any access granted by the allower client with client_id: {allower_id}."

        if allowed_model_id not in self._allowance[allower_id][allowee_id]:
            return False, f"The model with ml_model_id: {allowed_model_id} is not allowed for the allowee client with client_id: {allowee_id} by the allower client with client_id: {allower_id}."

        self._allowance[allower_id][allowee_id].remove(allowed_model_id)
        return True, None

    def get_allowed_models(self, allower_id, allowee_id):
        """
        Retrieve a list of machine learning model IDs that a client is allowed to access from another client.

        :param allower_id: The ID of the client who granted access.
        :type allower_id: str
        :param allowee_id: The ID of the client who has been granted access.
        :type allowee_id: str
        :return: A list of allowed machine learning model IDs or an error message if access is not granted.
        """
        if allower_id not in self._clients:
            return None, f"The allower client with client_id: {allower_id} doesn't exist."
        if allowee_id not in self._clients:
            return None, f"The allowee client with client_id: {allowee_id} doesn't exist."

        return self._allowance.get(allower_id).get(allowee_id, []), None

    def get_clients_allowance(self, client_id):
        """
        Retrieve the allowance dictionary for a given client.

        :param client_id: The ID of the client whose allowance is being retrieved.
        :type client_id: str
        :return: A dictionary containing the allowance information for the client.
        """
        if client_id not in self._allowance:
            return None, f"The client with client_id: {client_id} doesn't exist."
        return self._allowance[client_id], None

    def get_clients(self):
        """
        Retrieve a list of all registered client IDs.

        :return: A list of client IDs.
        """
        return [id for id in self._clients.keys()]

    def init_ml_model(self, client_id, ml_model_id):
        """
        Initialize a new machine learning model for a given client.

        :param client_id: The ID of the client to associate with the model.
        :type client_id: str
        :param ml_model_id: The ID of the machine learning model to initialize.
        :type ml_model_id: str
        :return: A tuple containing a boolean indicating success and an error message if the model already exists or the client ID is invalid.
        """
        if client_id not in self._clients:
            return False, "The given client_id is invalid."

        if ml_model_id in self._clients[client_id]:
            return False, f"The given ml_model_id: {ml_model_id} already exists within ml models of the client with client_id of {client_id}."

        self._clients[client_id][ml_model_id] = {}
        return True, None

    def set_ml_model(self, client_id, ml_model_id, ml_model):
        """
        Set or update the machine learning model for a given client.

        :param client_id: The ID of the client.
        :type client_id: str
        :param ml_model_id: The ID of the machine learning model.
        :type ml_model_id: str
        :param ml_model: The machine learning model object to be set.
        :type ml_model: obj
        :return: None
        """
        self._clients[client_id][ml_model_id] = ml_model

    def remove_ml_model(self, client_id, ml_model_id):
        """
        Remove an existing machine learning model for a given client.

        :param client_id: The ID of the client.
        :type client_id: str
        :param ml_model_id: The ID of the machine learning model to remove.
        :type ml_model_id: str
        :return: A tuple containing a boolean indicating success and an error message if the client ID or model ID is invalid.
        """
        if client_id not in self._clients:
            return False, "The given client_id is invalid."

        if ml_model_id not in self._clients[client_id]:
            return False, f"The client with client_id: {client_id} doesn't have any model with ml_model_id of {ml_model_id}."

        del self._clients[client_id][ml_model_id]
        return True, None

    def get_ml_models(self, client_id):
        """
        Retrieve a list of all machine learning model IDs associated with a given client.

        :param client_id: The ID of the client.
        :type client_id: str
        :return: A list of machine learning model IDs.
        """
        return [id for id in self._clients[client_id].keys()]
