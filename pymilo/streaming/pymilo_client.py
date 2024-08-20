# -*- coding: utf-8 -*-
"""PyMiloClient for RESTFull Protocol."""
from enum import Enum
from .encryptor import DummyEncryptor
from .compressor import get_compressor, Compression
from ..pymilo_obj import Export, Import
from .communicator import RESTClientCommunicator
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class Mode(Enum):
    """fallback state of the PyMiloClient."""

    LOCAL = 1
    DELEGATE = 2


class PymiloClient:
    """Facilitate working with the PyMilo server."""

    def __init__(
            self,
            model=None,
            mode=Mode.LOCAL,
            compressor=Compression.NONE,
            server="http://127.0.0.1",
            port=8000,
    ):
        """
        Initialize the Pymilo PymiloClient instance.

        :param model: the ML model PyMiloClient wrapped around
        :type model: Any
        :param mode: the mode in which PymiloClient should work, either LOCAL mode or DELEGATE
        :type mode: str (LOCAL|DELEGATE)
        :param compressor: the compression method to be used in client-server communications
        :type compressor: pymilo.streaming.compressor.Compression
        :param server: the url to which PyMilo Server listens
        :type server: str
        :param port: the port to which PyMilo Server listens
        :type port: int
        :return: an instance of the Pymilo PymiloClient class
        """
        self._client_id = "0x_client_id"
        self._model_id = "0x_model_id"
        self._model = model
        self._mode = mode
        self._compressor = get_compressor(compressor)
        self._encryptor = DummyEncryptor()
        self._communicator = RESTClientCommunicator(
            server_url="{}:{}".format(server, port)
        )

    def encrypt_compress(self, body):
        """
        Compress and Encrypt body payload.

        :param body: body payload of the request
        :type body: dict
        :return: the compressed and encrypted version of the body payload
        """
        return self._encryptor.encrypt(
            self._compressor.compress(
                body
            )
        )

    def toggle_mode(self, mode=Mode.LOCAL):
        """
        Toggle the PyMiloClient mode, either from LOCAL to DELEGATE or vice versa.

        :return: None
        """
        if mode not in Mode.__members__.values():
            raise Exception("Invalid mode, the given mode should be either `LOCAL`[default] or `DELEGATE`.")
        self._mode = mode

    def download(self):
        """
        Request for the remote ML model to download.

        :return: None
        """
        response = self._communicator.download(
            self.encrypt_compress(
                {
                    "client_id": self._client_id,
                    "model_id": self._model_id,
                }
            )
        )
        if response.status_code != 200:
            print("Remote model download failed.")
        print("Remote model downloaded successfully.")
        serialized_model = response.json()["payload"]
        self._model = Import(file_adr=None, json_dump=serialized_model).to_model()
        print("Local model updated successfully.")

    def upload(self):
        """
        Upload the local ML model to the remote server.

        :return: None
        """
        response = self._communicator.upload(
            self.encrypt_compress(
                {
                    "client_id": self._client_id,
                    "model_id": self._model_id,
                    "model": Export(self._model).to_json(),
                }
            )
        )
        if response.status_code == 200:
            print("Local model uploaded successfully.")
        else:
            print("Local model upload failed.")

    def __getattr__(self, attribute):
        """
        Overwrite the __getattr__ default function to extract requested.

            1. If self._mode is LOCAL, extract the requested from inner ML model and returns it
            2. If self._mode is DELEGATE, returns a wrapper relayer which delegates the request to the remote server by execution

        :return: Any
        """
        if self._mode == Mode.LOCAL:
            if attribute in dir(self._model):
                return getattr(self._model, attribute)
            else:
                raise AttributeError("This attribute doesn't exist in either PymiloClient or the inner ML model.")
        elif self._mode == Mode.DELEGATE:
            gdst = GeneralDataStructureTransporter()

            def relayer(*args, **kwargs):
                print(f"Method '{attribute}' called with args: {args} and kwargs: {kwargs}")
                payload = {
                    "client_id": self._client_id,
                    "model_id": self._model_id,
                    'attribute': attribute,
                    'args': args,
                    'kwargs': kwargs,
                }
                payload["args"] = gdst.serialize(payload, "args", None)
                payload["kwargs"] = gdst.serialize(payload, "kwargs", None)
                result = self._communicator.attribute_call(self.encrypt_compress(payload)).json()
                return gdst.deserialize(result, "payload", None)
            relayer.__doc__ = getattr(self._model.__class__, attribute).__doc__
            return relayer
