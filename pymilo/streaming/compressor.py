# -*- coding: utf-8 -*-
"""Implementations of Compressor interface."""
import gzip
import zlib
import lzma
import bz2
import json
import base64
from enum import Enum
from pymilo.streaming.interfaces import Compressor


class DummyCompressor(Compressor):
    """A dummy implementation of the Compressor interface."""

    @staticmethod
    def compress(payload):
        """Compress the given payload in a dummy way, simply just return it (no compression applied)."""
        return payload if isinstance(payload, str) else json.dumps(payload)

    @staticmethod
    def extract(payload):
        """Extract the given payload in a dummy way, simply just return it (no Extraction applied)."""
        return payload


class GZIPCompressor(Compressor):
    """GZIP implementation of the Compressor interface."""

    @staticmethod
    def compress(payload):
        """Compress the given payload using gzip."""
        if isinstance(payload, str):
            data = payload.encode('utf-8')
        else:
            data = json.dumps(payload).encode('utf-8')
        compressed_data = gzip.compress(data)
        return base64.b64encode(compressed_data).decode('utf-8')

    @staticmethod
    def extract(payload):
        """Extract the given payload using gzip."""
        data = base64.b64decode(payload)
        return gzip.decompress(data).decode('utf-8')


class ZLIBCompressor(Compressor):
    """ZLIB implementation of the Compressor interface."""

    @staticmethod
    def compress(payload):
        """Compress the given payload using zlib."""
        if isinstance(payload, str):
            data = payload.encode('utf-8')
        else:
            data = json.dumps(payload).encode('utf-8')
        compressed_data = zlib.compress(data)
        return base64.b64encode(compressed_data).decode('utf-8')

    @staticmethod
    def extract(payload):
        """Extract the given payload using zlib."""
        data = base64.b64decode(payload)
        return zlib.decompress(data).decode('utf-8')


class LZMACompressor(Compressor):
    """LZMA implementation of the Compressor interface."""

    @staticmethod
    def compress(payload):
        """Compress the given payload using lzma."""
        if isinstance(payload, str):
            data = payload.encode('utf-8')
        else:
            data = json.dumps(payload).encode('utf-8')
        compressed_data = lzma.compress(data)
        return base64.b64encode(compressed_data).decode('utf-8')

    @staticmethod
    def extract(payload):
        """Extract the given payload using lzma."""
        data = base64.b64decode(payload)
        return lzma.decompress(data).decode('utf-8')


class BZ2Compressor(Compressor):
    """BZ2 implementation of the Compressor interface."""

    @staticmethod
    def compress(payload):
        """Compress the given payload using bz2."""
        if isinstance(payload, str):
            data = payload.encode('utf-8')
        else:
            data = json.dumps(payload).encode('utf-8')
        compressed_data = bz2.compress(data)
        return base64.b64encode(compressed_data).decode('utf-8')

    @staticmethod
    def extract(payload):
        """Extract the given payload using bz2."""
        data = base64.b64decode(payload)
        return bz2.decompress(data).decode('utf-8')


class Compression(Enum):
    """Compression method used in end to end communication."""

    NULL = DummyCompressor
    GZIP = GZIPCompressor
    ZLIB = ZLIBCompressor
    LZMA = LZMACompressor
    BZ2 = BZ2Compressor
