from .compressor import DummyCompressor
from .encryptor import DummyEncryptor
from .communicator import DummyCommunicator


class PymiloClient:
    def __init__(self, model):
        self._model = model
        self.compressor = DummyCompressor()
        self.encryptor = DummyEncryptor()
        self.communicator = DummyCommunicator()

