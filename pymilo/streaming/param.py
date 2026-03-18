# -*- coding: utf-8 -*-
"""Streaming Parameters and constants."""
PYMILO_CLIENT_INVALID_MODE = "Invalid mode, the given mode should be either `LOCAL`[default] or `DELEGATE`."
PYMILO_CLIENT_MODEL_SYNCHED = "PyMiloClient synched the local ML model with the remote one successfully."
PYMILO_CLIENT_LOCAL_MODEL_UPLOADED = "PyMiloClient uploaded the local model successfully."
PYMILO_CLIENT_LOCAL_MODEL_UPLOAD_FAILED = "PyMiloClient failed to upload the local model."
PYMILO_CLIENT_INVALID_ATTRIBUTE = "This attribute doesn't exist in either PymiloClient or the inner ML model."
PYMILO_CLIENT_FAILED_TO_DOWNLOAD_REMOTE_MODEL = "PyMiloClient failed to download the remote ML model."

PYMILO_SERVER_NON_EXISTENT_ATTRIBUTE = "The requested attribute doesn't exist in this model."
PYMILO_INVALID_URL = "The given URL is not valid."
PYMILO_CLIENT_WEBSOCKET_NOT_CONNECTED = "WebSocket is not connected."

REST_API_PREFIX = "/api/v1"

MSG_DOWNLOAD_REQUEST = "Download request from client: {client_id} for model: {ml_model_id}"
MSG_UPLOAD_REQUEST = "Upload request from client: {client_id} for model: {ml_model_id}"
MSG_ATTRIBUTE_CALL_REQUEST = "Attribute call request from client: {client_id} for model: {ml_model_id}"
MSG_ATTRIBUTE_TYPE_REQUEST = "Attribute type request from client: {client_id} for model: {ml_model_id}"
MSG_REST_DOWNLOAD_REQUEST = "/download request from client: {client_id} for model: {ml_model_id}"
MSG_REST_UPLOAD_REQUEST = "/upload request from client: {client_id} for model: {ml_model_id}"
MSG_REST_ATTRIBUTE_CALL_REQUEST = "/attribute_call request from client: {client_id} for model: {ml_model_id}"
MSG_REST_ATTRIBUTE_TYPE_REQUEST = "/attribute_type request from client: {client_id} for model: {ml_model_id}"
