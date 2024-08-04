# -*- coding: utf-8 -*-
"""PyMilo ML Streaming Interfaces."""
from abc import ABC, abstractmethod


class Compressor(ABC):

    @abstractmethod
    def compress(payload):
        pass

    @abstractmethod
    def extract(payload):
        pass


class Encryptor(ABC):

    @abstractmethod
    def encrypt(payload):
        pass

    @abstractmethod
    def decrypt(payload):
        pass


class Communicator(ABC):

    @abstractmethod
    def upload(self, payload):
        pass

    @abstractmethod
    def download(self, payload):
        pass

    @abstractmethod
    def attribute_call(self, payload):
        pass
