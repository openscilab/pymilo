# -*- coding: utf-8 -*-
"""PyMilo Serialization Exception."""

from enum import Enum
from .pymilo_exception import PymiloException


class SerializationErrorTypes(Enum):
    """An enum class used to determine the type of serialization errors."""

    INVALID_MODEL = 1
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 2


class PymiloSerializationException(PymiloException):
    """
    Handle exceptions associated with Serializations.

    There are 2 different types of serialization exceptions:

        1-INVALID_MODEL: This error type claims that the given model is not a valid sklearn's linear model.

        2-VALID_MODEL_INVALID_INTERNAL_STRUCTURE: This error occurs when attempting to serialize a model that
        is one of the sklearn's linear models but it's internal structure has changed in a way that can't be serialized.
    """

    def __init__(self, meta_data):
        """
        Initialize the PymiloSerializationException instance.

        :param meta_data: Details pertain to the populated error.
        :type meta_data: dict [str:str]
        :return: an instance of the PymiloSerializationException class
        """
        # Call the base class constructor with the parameters it needs
        message = "Pymilo Serialization failed since "
        error_type = meta_data['error_type']
        error_type_to_message = {
            SerializationErrorTypes.INVALID_MODEL: 'the given model is not supported or is not a valid model.',
            SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE: 'the given model has some non-standard customized internal objects or functions.'}
        if error_type in error_type_to_message:
            message += error_type_to_message[error_type]
        else:
            message += "an Unknown error occurred."
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        """
        Generate a comprehensive report of the populated error.

        :return: error's details as dictionary
        """
        pymilo_report = super().to_pymilo_log()
        # TODO add any serializable field to `object` field of pymilo_report
        return pymilo_report
