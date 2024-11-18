# -*- coding: utf-8 -*-
"""ML Streaming utility module."""
import re


def validate_websocket_url(url: str) -> str:
    """
    Validate a WebSocket URL and add the 'ws://' protocol if missing.

    :param url: The WebSocket URL to validate.
    :type url: str
    :return: A tuple where the first element is a boolean indicating whether the URL is valid,
             and the second element is the possibly corrected URL (or None if invalid).
    """
    pattern = r"^(ws|wss)://"
    if not re.match(pattern, url):
        protocol = "ws://"
        url = protocol + url
    full_pattern = r"^(ws|wss)://([a-zA-Z0-9.-]+)(:[0-9]+)?(/.*)?$"
    if not re.match(full_pattern, url):
        return False, None
    return True, url


def validate_http_url(url: str) -> str:
    """
    Validate a HTTP URL and add the 'http://' protocol if missing.

    :param url: The HTTP URL to validate.
    :type url: str
    :return: A tuple where the first element is a boolean indicating whether the URL is valid,
             and the second element is the possibly corrected URL (or None if invalid).
    """
    pattern = r"^(http|https)://"
    if not re.match(pattern, url):
        protocol = "http://"
        url = protocol + url
    full_pattern = r"^(http|https)://([a-zA-Z0-9.-]+)(:[0-9]+)?(/.*)?$"
    if not re.match(full_pattern, url):
        return False, None
    return True, url
