# -*- coding: utf-8 -*-
"""PyMilo Serialization Exception."""

from enum import Enum
from .pymilo_exception import PymiloException


class SerilaizatoinErrorTypes(Enum):
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
        :return: an intance of the PymiloSerializationException class
        """
        # Call the base class constructor with the parameters it needs
        message = "Pymilo Serialization failed since "
        error_type = meta_data['error_type']
        error_type_to_message = {
            SerilaizatoinErrorTypes.INVALID_MODEL: 'the given model is not supported or is not a valid model.',
            SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE: 'the given model has some non-standard customized internal objects or functions.'}
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

    def to_pymilo_issue(self):
        """
        Generate an issue form from the populated error.

        :return: issue form of the associated error as string
        """
        pymilo_report = self.to_pymilo_log()
        help_request = """
        \n\nIn order to help us enhance Pymilo's functionality, please open an issue associated with this error and put the message below inside.\n
        """
        discription = "#### Description\n Pymilo Export failed."
        steps_to_produce = "\n#### Steps/Code to Reproduce\n It is auto-reported from the pymilo logger."
        expected_behaviour = "\n#### Expected Behavior\n A successfull Pymilo Export."
        actual_behaviour = "\n#### Actual Behavior\n Pymilo Export failed."
        operating_system = "#### Operating System\n {os}".format(
            os=pymilo_report['os']['full-description'])
        python_version = "#### Python Version\n {python_version}".format(
            python_version=pymilo_report['versions']["python-version"])
        pymilo_version = "#### PyMilo Version\n {pymilo_version}".format(
            pymilo_version=pymilo_report['versions']["pymilo-version"])
        gathered_data = "#### Logged Data\n {logged_data}".format(
            logged_data=str(pymilo_report))

        full_issue_form = help_request + discription + steps_to_produce + expected_behaviour + \
            actual_behaviour + operating_system + python_version + pymilo_version + gathered_data
        return full_issue_form
