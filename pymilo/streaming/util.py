# -*- coding: utf-8 -*-
"""ML Streaming utility module."""
import os
import re
from ..pymilo_param import URL_REGEX


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


def generate_dockerfile(
        dockerfile_name="Dockerfile",
        model_path=None,
        compression='NULL',
        protocol='REST',
        port=8000,
        init_model=None,
        bare=False
):
    """
    Generate a Dockerfile for running a PyMilo server with specified configurations.

    :param dockerfile_name: Name of the dockerfile.
    :type dockerfile_name: str
    :param model_path: Path or URL to the exported model JSON file.
    :type model_path: str
    :param compression: Compression method (default: NULL).
    :type compression: str
    :param protocol: Communication protocol (default: REST).
    :type protocol: str
    :param port: Port for the PyMilo server (default: 8000).
    :type port: int
    :param init_model: The model that the server initialized with.
    :type init_model: boolean
    :param bare: A flag that sets if the server runs without an internal ML model.
    :type bare: boolean
    """
    dockerfile_content = f"""# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install pymilo
RUN pip install pymilo[streaming]
    """
    is_url = False
    if model_path:
        if re.match(URL_REGEX, model_path):
            is_url = True
        else:
            dockerfile_content += f"\nCOPY {os.path.basename(model_path)} /app/model.json"

    # Expose the specified port
    dockerfile_content += f"\nEXPOSE {port}"

    cmd = "CMD [\"python\", \"-m\", \"pymilo\""
    cmd += f", \"--compression\", \"{compression}\""
    cmd += f", \"--protocol\", \"{protocol}\""
    cmd += f", \"--port\", \"{port}\""

    if model_path:
        if is_url:
            cmd += f", \"--load\", \"{model_path}\""
        else:
            cmd += f", \"--load\", \"/app/model.json\""
    elif init_model:
        cmd += f", \"--init\" \"{init_model}\""
    elif bare:
        cmd += f", \"--bare\""

    cmd += "]"
    dockerfile_content += f"\n{cmd}\n"

    with open(dockerfile_name, 'w') as f:
        f.write(dockerfile_content)
