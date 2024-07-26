from .interfaces import Encryptor


class DummyEncryptor(Encryptor):
    def encrypt(string):
        return string 
    def decrypt(string):
        return string