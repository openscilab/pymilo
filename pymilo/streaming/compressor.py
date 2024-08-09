from .interfaces import Compressor


class DummyCompressor(Compressor):

    @staticmethod
    def compress(payload):
        return payload

    @staticmethod
    def extract(payload):
        return payload
