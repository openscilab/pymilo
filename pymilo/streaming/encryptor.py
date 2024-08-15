# -*- coding: utf-8 -*-
"""Implementations of Encryptor interface."""
from .interfaces import Encryptor


class DummyEncryptor(Encryptor):
    """The Pymilo DummyEncryptor class is a dummy implementation of the Encryptor interface to act as a simple wire."""

    @staticmethod
    def encrypt(payload):
        """Encrypt the given payload in a dummy way, simply just return it (no encryption applied)."""
        return payload

    @staticmethod
    def decrypt(payload):
        """Decrypt the given payload in a dummy way, simply just return it (no decryption applied)."""
        return payload
