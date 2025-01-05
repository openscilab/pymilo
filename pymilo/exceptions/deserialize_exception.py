# -*- coding: utf-8 -*-
"""PyMilo Deserialization Exception."""
from enum import Enum
from .pymilo_exception import PymiloException


class DeserializationErrorTypes(Enum):
    """An enum class to determine the type of deserialization errors."""

    CORRUPTED_JSON_FILE = 1
    INVALID_MODEL = 2
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3


class PymiloDeserializationException(PymiloException):
    """
    Handle exceptions associated with Deserialization.

    There are 3 different types of deserialization exceptions:

        1-CORRUPTED_JSON_FILE: This error type claims that the given json string file which is supposed to be an
        output of Pymilo Export, is corrupted and can not be parsed as a valid json.

        2-INVALID_MODEL: This error type claims that the given json string file(or object) is not a deserialized export of
        a valid sklearn linear model.

        3-VALID_MODEL_INVALID_INTERNAL_STRUCTURE: This error occurs when attempting to load a JSON file or object that
        does not conform to the expected format of a serialized scikit-learn linear model.
        The file may have been modified after being exported from Pymilo Export, causing it to become invalid.
    """

    def __init__(self, meta_data):
        """
        Initialize the PymiloDeserializationException instance.

        :param meta_data: Details pertain to the populated error.
        :type meta_data: dict [str:str]
        :return: an instance of the PymiloDeserializationException class
        """
        # Call the base class constructor with the parameters it needs
        message = "Pymilo Deserialization failed since {reason}"
        error_type = meta_data['error_type']
        error_type_to_message = {
            DeserializationErrorTypes.CORRUPTED_JSON_FILE:
            'the given file is not a valid .json file.',
            DeserializationErrorTypes.INVALID_MODEL:
            'the given model is not supported or is not a valid model.',
            DeserializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE:
            'the given model has some non-standard customized internal objects or functions.'}
        if error_type in error_type_to_message:
            reason = error_type_to_message[error_type]
        else:
            reason = "an Unknown error occurred."
        message.format(reason=reason)
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        """
        Generate a comprehensive report of the populated error.

        :return: a dictionary of error details.
        """
        pymilo_report = super().to_pymilo_log()
        if self.meta_data['error_type'] == DeserializationErrorTypes.CORRUPTED_JSON_FILE:
            pymilo_report['object']['json_file'] = self.meta_data['json_file']
        return pymilo_report
