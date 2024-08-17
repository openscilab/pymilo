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
        :return: the compressed version
        """

    @abstractmethod
    def extract(payload):
        """
        Extract the given previously compressed payload.

        :param payload: payload to get extracted
        :type payload: str
        :return: the extracted version
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
        :return: the encrypted version
        """

    @abstractmethod
    def decrypt(payload):
        """
        Decrypt the given previously encrypted payload.

        :param payload: payload to get decrypted
        :type payload: str
        :return: the decrypted version
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
        :return: remote server response
        """

    @abstractmethod
    def download(self, payload):
        """
        Download the remote ML model to local.

        :param payload: request payload
        :type payload: dict
        :return: remote server response
        """

    @abstractmethod
    def attribute_call(self, payload):
        """
        Execute an attribute call on the remote server.

        :param payload: request payload
        :type payload: dict
        :return: remote server response
        """
