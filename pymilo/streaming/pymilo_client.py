# -*- coding: utf-8 -*-
"""PyMiloClient for RESTFull Protocol."""
from enum import Enum
from .encryptor import DummyEncryptor
from .compressor import DummyCompressor
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
            server_url="http://127.0.0.1:8000",
    ):
        """
        Initialize the Pymilo PymiloClient instance.

        :param model: the ML model PyMiloClient wrapped around
        :type model: Any
        :param mode: the mode in which PymiloClient should work, either LOCAL mode or DELEGATE
        :type mode: str (LOCAL|DELEGATE)
        :param server_url: the url to which PyMilo Server listens
        :type server_url: str
        :return: an instance of the Pymilo PymiloClient class
        """
        self._client_id = "0x_client_id"
        self._model_id = "0x_model_id"
        self._model = model
        self._mode = mode
        self._compressor = DummyCompressor()
        self._encryptor = DummyEncryptor()
        self._communicator = RESTClientCommunicator(server_url)

    def toggle_mode(self, mode=Mode.LOCAL):
        """
        Toggle the PyMiloClient mode, either from LOCAL to DELEGATE or vice versa.

        :return: None
        """
        if mode not in Mode.__members__.values():
            raise Exception("Invalid mode, the given mode should be either `LOCAL`[default] or `DELEGATE`.")
        if mode != self._mode:
            self._mode = mode

    def download(self):
        """
        Request for the remote ML model to download.

        :return: None
        """
        serialized_model = self._communicator.download({
            "client_id": self._client_id,
            "model_id": self._model_id
        })
        if serialized_model is None:
            print("PyMiloClient failed to download the remote ML model.")
            return
        self._model = Import(file_adr=None, json_dump=serialized_model).to_model()
        print("PyMiloClient synched the local ML model with the remote one successfully.")

    def upload(self):
        """
        Upload the local ML model to the remote server.

        :return: None
        """
        succeed = self._communicator.upload({
            "client_id": self._client_id,
            "model_id": self._model_id,
            "model": Export(self._model).to_json(),
        })
        if succeed:
            print("PyMiloClient uploaded the local model successfully.")
        else:
            print("PyMiloClient failed to upload the local model.")

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
                payload = {
                    "client_id": self._client_id,
                    "model_id": self._model_id,
                    'attribute': attribute,
                    'args': args,
                    'kwargs': kwargs,
                }
                payload["args"] = gdst.serialize(payload, "args", None)
                payload["kwargs"] = gdst.serialize(payload, "kwargs", None)
                result = self._communicator.attribute_call(
                    self._encryptor.encrypt(
                        self._compressor.compress(
                            payload
                        )
                    )
                )
                return gdst.deserialize(result, "payload", None)
            relayer.__doc__ = getattr(self._model.__class__, attribute).__doc__
            return relayer
