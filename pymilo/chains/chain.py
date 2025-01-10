# -*- coding: utf-8 -*-
"""PyMilo Chain Module."""

from traceback import format_exc
from abc import ABC, abstractmethod

from ..utils.util import get_sklearn_type
from ..transporters.transporter import Command
from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes


class Chain(ABC):
    """
    Chain Interface.

    Each Chain serializes/deserializes the given model.
    """

    @abstractmethod
    def is_supported(self, model):
        """
        Check if the given model is a sklearn's ML model supported by this chain.

        :param model: a string name of an ML model or a sklearn object of it
        :type model: any object
        :return: check result as bool
        """

    @abstractmethod
    def transport(self, request, command, is_inner_model=False):
        """
        Return the transported (serialized or deserialized) model.

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
    """Abstract Chain with the general implementation of the Chain interface."""

    def __init__(self, transporters, supported_models):
        """
        Initialize the AbstractChain instance.

        :param transporters: worker transporters dedicated to this chain
        :type transporters: transporter.AbstractTransporter[]
        :param supported_models: supported sklearn ML models belong to this chain
        :type supported_models: dict
        :return: an instance of the AbstractChain class
        """
        self._transporters = transporters
        self._supported_models = supported_models

    def is_supported(self, model):
        """
        Check if the given model is a sklearn's ML model supported by this chain.

        :param model: a string name of an ML model or a sklearn object of it
        :type model: any object
        :return: check result as bool
        """
        model_name = model if isinstance(model, str) else get_sklearn_type(model)
        return model_name in self._supported_models

    def transport(self, request, command, is_inner_model=False):
        """
        Return the transported (serialized or deserialized) model.

        :param request: given ML model to be transported
        :type request: any object
        :param command: command to specify whether the request should be serialized or deserialized
        :type command: transporter.Command
        :param is_inner_model: determines whether it is an inner model of a super ML model
        :type is_inner_model: boolean
        :return: the transported request as a json string or sklearn ML model
        """
        if not is_inner_model:
            self.validate(request, command)

        if command == Command.SERIALIZE:
            try:
                return self.serialize(request)
            except Exception as e:
                raise PymiloSerializationException(
                    {
                        'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                        'error': {
                            'Exception': repr(e),
                            'Traceback': format_exc(),
                        },
                        'object': request,
                    })

        elif command == Command.DESERIALIZE:
            try:
                return self.deserialize(request, is_inner_model)
            except Exception as e:
                raise PymiloDeserializationException(
                    {
                        'error_type': DeserializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                        'error': {
                            'Exception': repr(e),
                            'Traceback': format_exc()},
                        'object': request
                    })

    def serialize(self, model):
        """
        Return the serialized json string of the given model.

        :param model: given ML model to be get serialized
        :type model: sklearn ML model
        :return: the serialized json string of the given ML model
        """
        for transporter in self._transporters:
            self._transporters[transporter].transport(model, Command.SERIALIZE)
        return model.__dict__

    def deserialize(self, serialized_model, is_inner_model=False):
        """
        Return the associated sklearn ML model of the given previously serialized ML model.

        :param serialized_model: given json string of a ML model to get deserialized to associated sklearn ML model
        :type serialized_model: obj
        :param is_inner_model: determines whether it is an inner ML model of a super ML model
        :type is_inner_model: boolean
        :return: associated sklearn ML model
        """
        raw_model = None
        data = None
        if is_inner_model:
            raw_model = self._supported_models[serialized_model["type"]]()
            data = serialized_model["data"]
        else:
            raw_model = self._supported_models[serialized_model.type]()
            data = serialized_model.data
        for transporter in self._transporters:
            self._transporters[transporter].transport(
                serialized_model, Command.DESERIALIZE, is_inner_model)
        for item in data:
            setattr(raw_model, item, data[item])
        return raw_model

    def validate(self, model, command):
        """
        Check if the provided inputs are valid in relation to each other.

        :param model: a sklearn ML model or a json string of it, serialized through the pymilo export
        :type model: obj
        :param command: command to specify whether the request should be serialized or deserialized
        :type command: transporter.Command
        :return: None
        """
        if command == Command.SERIALIZE:
            if self.is_supported(model):
                return
            else:
                raise PymiloSerializationException(
                    {
                        'error_type': SerializationErrorTypes.INVALID_MODEL,
                        'object': model
                    }
                )
        elif command == Command.DESERIALIZE:
            if self.is_supported(model.type):
                return
            else:
                raise PymiloDeserializationException(
                    {
                        'error_type': DeserializationErrorTypes.INVALID_MODEL,
                        'object': model
                    }
                )
