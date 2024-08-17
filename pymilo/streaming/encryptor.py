# -*- coding: utf-8 -*-
"""Implementations of Encryptor interface."""
from .interfaces import Encryptor


class DummyEncryptor(Encryptor):
    """A dummy implementation of the Encryptor interface."""

    @staticmethod
    def encrypt(payload):
        """Encrypt the given payload in a dummy way, simply just return it (no encryption applied)."""
        return payload

    @staticmethod
    def decrypt(payload):
        """Decrypt the given payload in a dummy way, simply just return it (no decryption applied)."""
        return payload
