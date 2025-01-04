# -*- coding: utf-8 -*-
"""PyMilo Chain Module"""
from traceback import format_exc
from abc import ABC, abstractmethod

from ..utils.util import get_sklearn_type
from ..transporters.transporter import Command
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes


class Chain(ABC):
    """
    Chain Interface.

    Each Chain transports(either serializes or deserializes) the given model according to the given command.
    """

    @abstractmethod
    def is_supported(self, model):
        """
        Check if the given model is a sklearn's ML model supported by this chain.

        :param model: is a string name of a ML model or a sklearn object of it
        :type model: any object
        :return: check result as bool
        """

    @abstractmethod
    def transport(self, request, command, is_inner_model=False):
        """
        Return the transported (Serialized or Deserialized) model.

        :param request: given ML model to be transported
        :type request: any object
        :param command: command to specify whether the request should be serialized or deserialized
        :type command: transporter.Command
        :param is_inner_model: determines whether it is an inner model of a super ML model
        :type is_inner_model: boolean
        :return: the transported request as a json string or sklearn ML model
        """

    @abstractmethod
    def serialize(self, model):
        """
        Return the serialized json string of the given model.

        basically in order to fully serialize a model, we should pass it through the chain of associated
        transporters to get fully serialized.

        :param model: given ML model to be get serialized
        :type model: sklearn ML model
        :return: the serialized json string of the given ML model
        """

    @abstractmethod
    def deserialize(self, serialized_model, is_inner_model=False):
        """
        Return the associated sklearn ML model of the given previously serialized ML model.

        :param serialized_model: given json string of a ML model to get deserialized to associated sklearn ML model
        :type serialized_model: obj
        :param is_inner_model: determines whether it is an inner ML model of a super ML model
        :type is_inner_model: boolean
        :return: associated sklearn ML model
        """

    @abstractmethod
    def validate(self, model, command):
        """
        Check if the provided inputs are valid in relation to each other.

        :param model: a sklearn ML model or a json string of it, serialized through the pymilo export
        :type model: obj
        :param command: command to specify whether the request should be serialized or deserialized
        :type command: transporter.Command
        :return: None
        """


class AbstractChain(Chain):
