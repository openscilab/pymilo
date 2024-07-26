# -*- coding: utf-8 -*-
"""PyMilo ML Streaming Interfaces."""
from abc import ABC, abstractmethod


class Compressor(ABC):
    @abstractmethod
    def compress(string):
        pass
    @abstractmethod
    def extract(string):
        pass


class Encryptor(ABC):
    @abstractmethod
    def encrypt(string):
        pass
    @abstractmethod
    def decrypt(string):
        pass


class Communicator(ABC):
    @abstractmethod
    def send(self, string):
        pass
    @abstractmethod
    def receive(self, string):
        pass
