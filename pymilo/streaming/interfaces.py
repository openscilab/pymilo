# -*- coding: utf-8 -*-
"""PyMilo ML Streaming Interfaces."""
from abc import ABC, abstractmethod


class Compressor(ABC):

    @abstractmethod
    def compress(payload):
        """
        """

    @abstractmethod
    def extract(payload):
        """
        """


class Encryptor(ABC):

    @abstractmethod
    def encrypt(payload):
        """
        """

    @abstractmethod
    def decrypt(payload):
        """
        """


class ClientCommunicator(ABC):

    @abstractmethod
    def upload(self, payload):
        """
        """

    @abstractmethod
    def download(self, payload):
        """
        """

    @abstractmethod
    def attribute_call(self, payload):
        """
        """
