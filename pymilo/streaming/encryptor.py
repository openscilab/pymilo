from .interfaces import Encryptor


class DummyEncryptor(Encryptor):

    @staticmethod
    def encrypt(payload):
        return payload

    @staticmethod
    def decrypt(payload):
        return payload
