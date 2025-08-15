# -*- coding: utf-8 -*-
"""PyMiloClient for RESTFull Protocol."""
from enum import Enum
from .encryptor import DummyEncryptor
from .compressor import Compression
from ..pymilo_obj import Export, Import
from .param import PYMILO_CLIENT_INVALID_MODE, PYMILO_CLIENT_MODEL_SYNCHED, \
    PYMILO_CLIENT_LOCAL_MODEL_UPLOADED, PYMILO_CLIENT_LOCAL_MODEL_UPLOAD_FAILED, \
    PYMILO_CLIENT_INVALID_ATTRIBUTE, PYMILO_CLIENT_FAILED_TO_DOWNLOAD_REMOTE_MODEL
from .communicator import CommunicationProtocol
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class PymiloClient:
    """Facilitate working with the PyMilo server."""

    class Mode(Enum):
        """fallback state of the PyMiloClient."""

        LOCAL = 1
        DELEGATE = 2

    def __init__(
            self,
            model=None,
            mode=Mode.LOCAL,
            compressor=Compression.NULL,
            server_url="127.0.0.1:8000",
            communication_protocol=CommunicationProtocol.REST,
    ):
        """
        Initialize the Pymilo PymiloClient instance.

        :param model: the ML model PyMiloClient wrapped around
        :type model: Any
        :param mode: the mode in which PymiloClient should work, either LOCAL mode or DELEGATE
        :type mode: str (LOCAL|DELEGATE)
        :param compressor: the compression method to be used in client-server communications
        :type compressor: pymilo.streaming.compressor.Compression
        :param server_url: the url to which PyMilo Server listens
        :type server_url: str
        :param communication_protocol: The communication protocol to be used by PymiloClient
        :type communication_protocol: pymilo.streaming.communicator.CommunicationProtocol
        :return: an instance of the Pymilo PymiloClient class
        """
        self.model = model
        self.client_id = "0x_client_id"
        self.ml_model_id = "0x_ml_model_id"
        self._mode = mode
        self._compressor = compressor.value
        self._encryptor = DummyEncryptor()
        self._communicator = communication_protocol.value["CLIENT"](server_url)

    def encrypt_compress(self, body):
        """
        Compress and Encrypt body payload.

        :param body: body payload of the request
        :type body: dict
        :return: the compressed and encrypted version of the body payload
        """
        return self._encryptor.encrypt(
            self._compressor.compress(body)
        )

    def toggle_mode(self, mode=Mode.LOCAL):
        """
        Toggle the PyMiloClient mode, either from LOCAL to DELEGATE or vice versa.

        :return: None
        """
        if mode not in PymiloClient.Mode.__members__.values():
            raise Exception(PYMILO_CLIENT_INVALID_MODE)
        if mode != self._mode:
            self._mode = mode

    def download(self):
        """
        Request for the remote ML model to download.

        :return: None
        """
        serialized_model = self._communicator.download(
            self.client_id,
            self.ml_model_id
        )
        if serialized_model is None:
            print(PYMILO_CLIENT_FAILED_TO_DOWNLOAD_REMOTE_MODEL)
            return
        self.model = Import(file_adr=None, json_dump=serialized_model).to_model()
        print(PYMILO_CLIENT_MODEL_SYNCHED)

    def upload(self):
        """
        Upload the local ML model to the remote server.

        :return: None
        """
        succeed = self._communicator.upload(
            self.client_id,
            self.ml_model_id,
            self.encrypt_compress({"model": Export(self.model).to_json()})
        )
        if succeed:
            print(PYMILO_CLIENT_LOCAL_MODEL_UPLOADED)
        else:
            print(PYMILO_CLIENT_LOCAL_MODEL_UPLOAD_FAILED)

    def register(self):
        """
        Register client in the remote server.

        :return: None
        """
        self.client_id = self._communicator.register_client()

    def deregister(self):
        """
        Deregister client in the remote server.

        :return: None
        """
        self._communicator.remove_client(self.client_id)
        self.client_id = "0x_client_id"

    def register_ml_model(self):
        """
        Register ML model in the remote server.

        :return: None
        """
        self.ml_model_id = self._communicator.register_model(self.client_id)

    def deregister_ml_model(self):
        """
        Deregister ML model in the remote server.

        :return: None
        """
        self._communicator.remove_model(self.client_id, self.ml_model_id)
        self.ml_model_id = "0x_ml_model_id"

    def get_ml_models(self):
        """
        Get all registered ml models in the remote server for this client.

        :return: list of ml model ids
        """
        return self._communicator.get_ml_models(self.client_id)

    def grant_access(self, allowee_id):
        """
        Grant access to one of this client's models to another client.

        :param allowee_id: The client ID to grant access to
        :return: True if successful, False otherwise
        """
        return self._communicator.grant_access(
            self.client_id,
            allowee_id,
            self.ml_model_id
        )

    def revoke_access(self, revokee_id):
        """
        Revoke access previously granted to another client.

        :param revokee_id: The client ID to revoke access from
        :return: True if successful, False otherwise
        """
        return self._communicator.revoke_access(
            self.client_id,
            revokee_id,
            self.ml_model_id
        )

    def get_allowance(self):
        """
        Get a dictionary of all clients who have access to this client's models.

        :return: Dict of allowee_id -> list of model_ids
        """
        return self._communicator.get_allowance(self.client_id)

    def get_allowed_models(self, allower_id):
        """
        Get a list of models you are allowed to access from another client.

        :param allower_id: The client ID who owns the models
        :return: list of allowed model IDs
        """
        return self._communicator.get_allowed_models(allower_id, self.client_id)

    def __getattr__(self, attribute):
        """
        Overwrite the __getattr__ default function to extract requested.

            1. If self._mode is LOCAL, extract the requested from inner ML model and returns it
            2. If self._mode is DELEGATE, returns a wrapper relayer which delegates the request to the remote server by execution

        :return: Any
        """
        if self._mode == PymiloClient.Mode.LOCAL:
            if attribute in dir(self.model):
                return getattr(self.model, attribute)
            else:
                raise AttributeError(PYMILO_CLIENT_INVALID_ATTRIBUTE)
        elif self._mode == PymiloClient.Mode.DELEGATE:
            gdst = GeneralDataStructureTransporter()
            response = self._communicator.attribute_type(
                self.client_id,
                self.ml_model_id,
                self.encrypt_compress({
                    "attribute": attribute,
                    "client_id": self.client_id,
                    "ml_model_id": self.ml_model_id,
                })
            )
            if response["attribute type"] == "field":
                return gdst.deserialize(response, "attribute value", None)

            def relayer(*args, **kwargs):
                payload = {
                    "client_id": self.client_id,
                    "ml_model_id": self.ml_model_id,
                    'attribute': attribute,
                    'args': args,
                    'kwargs': kwargs,
                }
                payload["args"] = gdst.serialize(payload, "args", None)
                payload["kwargs"] = gdst.serialize(payload, "kwargs", None)
                result = self._communicator.attribute_call(
                    self.client_id,
                    self.ml_model_id,
                    self.encrypt_compress(payload)
                )
                return gdst.deserialize(result, "payload", None)
            return relayer
