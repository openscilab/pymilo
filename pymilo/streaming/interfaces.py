# -*- coding: utf-8 -*-
"""PyMilo ML Streaming Interfaces."""
from abc import ABC, abstractmethod


class Compressor(ABC):
    """
    Compressor Interface.

    Each Compressor has methods to compress the given payload or extract it back to the original one.
    """

    @abstractmethod
    def compress(payload):
        """
        Compress the given payload.

        :param payload: payload to get compressed
        :type payload: str
        :return: str (compressed payload)
        """

    @abstractmethod
    def extract(payload):
        """
        Extract the given previously compressed payload.

        :param payload: payload to get extracted
        :type payload: str
        :return: str (extracted payload)
        """


class Encryptor(ABC):
    """
    Encryptor Interface.

    Each Encryptor has methods to encrypt the given payload or decrypt it back to the original one.
    """

    @abstractmethod
    def encrypt(payload):
        """
        Encrypt the given payload.

        :param payload: payload to get encrypted
        :type payload: str
        :return: str (encrypted payload)
        """

    @abstractmethod
    def decrypt(payload):
        """
        Decrypt the given previously encrypted payload.

        :param payload: payload to get decrypted
        :type payload: str
        :return: str (decrypted payload)
        """


class ClientCommunicator(ABC):
    """
    ClientCommunicator Interface.

    Each ClientCommunicator has methods to upload the local ML model, download the remote ML model and delegate attribute call to the remote server.
    """

    @abstractmethod
    def upload(self, payload):
        """
        Upload the given payload to the remote server.

        :param payload: request payload
        :type payload: dict
        :return: Response object (varies based on the Implemented Protocol)
        """

    @abstractmethod
    def download(self, payload):
        """
        Download the remote ML model to local.

        :param payload: request payload
        :type payload: dict
        :return: Response object (varies based on the Implemented Protocol)
        """

    @abstractmethod
    def attribute_call(self, payload):
        """
        Execute an attribute call on the remote server.

        :param payload: request payload
        :type payload: dict
        :return: Response object (varies based on the Implemented Protocol)
        """
