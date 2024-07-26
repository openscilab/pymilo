from .interfaces import Compressor


class DummyCompressor(Compressor):
    def compress(string):
        return string 
    def extract(string):
        return string