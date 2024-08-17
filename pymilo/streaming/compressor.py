# -*- coding: utf-8 -*-
"""Implementations of Compressor interface."""
from .interfaces import Compressor


class DummyCompressor(Compressor):
    """A dummy implementation of the Compressor interface."""

    @staticmethod
    def compress(payload):
        """Compress the given payload in a dummy way, simply just return it (no compression applied)."""
        return payload

    @staticmethod
    def extract(payload):
        """Extract the given payload in a dummy way, simply just return it (no Extraction applied)."""
        return payload
