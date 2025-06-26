# -*- coding: utf-8 -*-
"""PyMilo ML Streaming Interfaces."""
from abc import ABC, abstractmethod


class Compressor(ABC):
    """
    Compressor Interface.

    Each Compressor has methods to compress the given payload or extract it back to the original one.
    """

    @abstractmethod
    def compress(payload):
        """
        Compress the given payload.

        :param payload: payload to get compressed
        :type payload: str
        :return: the compressed version
        """

    @abstractmethod
    def extract(payload):
        """
        Extract the given previously compressed payload.

        :param payload: payload to get extracted
        :type payload: str
        :return: the extracted version
        """


class Encryptor(ABC):
    """
    Encryptor Interface.

    Each Encryptor has methods to encrypt the given payload or decrypt it back to the original one.
    """

    @abstractmethod
    def encrypt(payload):
        """
        Encrypt the given payload.

        :param payload: payload to get encrypted
        :type payload: str
        :return: the encrypted version
        """

    @abstractmethod
    def decrypt(payload):
        """
        Decrypt the given previously encrypted payload.

        :param payload: payload to get decrypted
        :type payload: str
        :return: the decrypted version
        """


class ClientCommunicator(ABC):
    """
    ClientCommunicator Interface.

    Defines the contract for client-server communication. Each implementation is responsible for:
    - Registering and removing clients and models
    - Uploading and downloading ML models
    - Handling delegated attribute access
    - Managing model allowances between clients
    """

    @abstractmethod
    def register_client(self):
        """
        Register the client in the remote server.

        :return: newly allocated client ID
        :rtype: str
        """

    @abstractmethod
    def remove_client(self, client_id):
        """
        Remove the client from the remote server.

        :param client_id: client ID to remove
        :type client_id: str
        :return: success status
        :rtype: bool
        """

    @abstractmethod
    def register_model(self, client_id):
        """
        Register an ML model for the given client.

        :param client_id: client ID
        :type client_id: str
        :return: newly allocated model ID
        :rtype: str
        """

    @abstractmethod
    def remove_model(self, client_id, model_id):
        """
        Remove the specified ML model for the client.

        :param client_id: client ID
        :type client_id: str
        :param model_id: model ID
        :type model_id: str
        :return: success status
        :rtype: bool
        """

    @abstractmethod
    def get_ml_models(self, client_id):
        """
        Get the list of ML models for the given client.

        :param client_id: client ID
        :type client_id: str
        :return: list of model IDs
        :rtype: list[str]
        """

    @abstractmethod
    def grant_access(self, allower_id, allowee_id, model_id):
        """
        Grant access to a model from one client to another.

        :param allower_id: client who owns the model
        :type allower_id: str
        :param allowee_id: client to be granted access
        :type allowee_id: str
        :param model_id: model ID
        :type model_id: str
        :return: success status
        :rtype: bool
        """

    @abstractmethod
    def revoke_access(self, revoker_id, revokee_id, model_id):
        """
        Revoke model access from one client to another.

        :param revoker_id: client who owns the model
        :type revoker_id: str
        :param revokee_id: client to be revoked
        :type revokee_id: str
        :param model_id: model ID
        :type model_id: str
        :return: success status
        :rtype: bool
        """

    @abstractmethod
    def get_allowance(self, allower_id):
        """
        Get all clients and models this client has allowed.

        :param allower_id: client who granted access
        :type allower_id: str
        :return: dictionary mapping allowee_id to list of model_ids
        :rtype: dict
        """

    @abstractmethod
    def get_allowed_models(self, allower_id, allowee_id):
        """
        Get the list of model IDs that `allowee_id` is allowed to access from `allower_id`.

        :param allower_id: model owner
        :type allower_id: str
        :param allowee_id: recipient
        :type allowee_id: str
        :return: list of allowed model IDs
        :rtype: list[str]
        """

    @abstractmethod
    def upload(self, client_id, model_id, model):
        """
        Upload the local ML model to the remote server.

        :param client_id: ID of the client
        :param model_id: ID of the model
        :param model: serialized model content
        :return: True if upload was successful, False otherwise
        """

    @abstractmethod
    def download(self, client_id, model_id):
        """
        Download the remote ML model.

        :param client_id: ID of the requesting client
        :param model_id: ID of the model to download
        :return: string serialized model
        """

    @abstractmethod
    def attribute_call(self, client_id, model_id, call_payload):
        """
        Execute an attribute call on the remote server.

        :param client_id: ID of the client
        :param model_id: ID of the model
        :param call_payload: payload containing attribute name, args, and kwargs
        :return: remote server response
        """

    @abstractmethod
    def attribute_type(self, client_id, model_id, type_payload):
        """
        Identify the attribute type (method or field) on the remote model.

        :param client_id: client ID
        :param model_id: model ID
        :param type_payload: payload containing targeted attribute
        :return: remote server response
        """
