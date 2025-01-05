# -*- coding: utf-8 -*-
"""PyMilo Abstract Exception Class."""

import pymilo
import sklearn
import platform
from datetime import datetime
from abc import ABC, abstractmethod


class PymiloException(Exception, ABC):
    """An abstract class for handling pymilo associated exceptions."""

    def __init__(self, message, meta_data):
        """
        Initialize the PymiloException instance.

        :param message: Error message associated with the populated error.
        :type message: str
        :param meta_data: Details pertain to the populated error.
        :type meta_data: dict [str:str]
        :return: an instance of the PymiloDeserializationException class
        """
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # gathered meta_data
        self.message = message
        self.meta_data = meta_data

    def to_pymilo_log(self):
        """
        Generate a comprehensive report of the populated error.

        :return: error's details as dictionary
        """
        pymilo_report = {
            'os': {
                'name': platform.system(),
                'version': platform.version(),
                'release': platform.release(),
                'full-description': platform.platform()
            },
            'versions': {
                'pymilo-version': pymilo.__version__,
                'scikit-version': sklearn.__version__,
                'python-version': platform.python_version()
            },
            'object': {
                'type': type(self.meta_data['object']),
                'content': self.meta_data['object']
            },
            'error': {
                'date-utc': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
                'pymilo-error': self.message,
                'inner-error': self.meta_data['error'] if "error" in self.meta_data else ""
            }
        }

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
        associated_pymilo_class = "Export" if "Serialization" in self.message else "Import"
        description = "#### Description\n Pymilo {pymilo_class} failed.".format(pymilo_class=associated_pymilo_class)
        steps_to_produce = "\n#### Steps/Code to Reproduce\n It is auto-reported from the pymilo logger."
        expected_behavior = "\n#### Expected Behavior\n A successful Pymilo {pymilo_class}.".format(
            pymilo_class=associated_pymilo_class)
        actual_behavior = "\n#### Actual Behavior\n Pymilo {pymilo_class} failed.".format(
            pymilo_class=associated_pymilo_class)
        operating_system = "#### Operating System\n {os}".format(os=pymilo_report['os']['full-description'])
        python_version = "#### Python Version\n {python_version}".format(
            python_version=pymilo_report['versions']["python-version"])
        pymilo_version = "#### PyMilo Version\n {pymilo_version}".format(
            pymilo_version=pymilo_report['versions']["pymilo-version"])
        gathered_data = "#### Logged Data\n {logged_data}".format(logged_data=str(pymilo_report))

        full_issue_form = help_request + description + steps_to_produce + expected_behavior + \
            actual_behavior + operating_system + python_version + pymilo_version + gathered_data
        return full_issue_form

    def __str__(self):
        """
        Override the base __str__ function.

        :return: issue form of the associated error as string
        """
        return self.to_pymilo_issue()
