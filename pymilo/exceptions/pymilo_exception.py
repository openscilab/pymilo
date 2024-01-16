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
        :return: an intance of the PymiloDeserializationException class
        """
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # gathered meta_data
        self.message = message
        self.meta_data = meta_data

    # collect All pymilo related data.
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

    @abstractmethod
    def to_pymilo_issue(self):
        """
        Generate an issue form from the populated error.

        :return: issue form of the associated error as string
        """

    def __str__(self):
        """
        Override the base __str__ function.

        :return: issue form of the associated error as string
        """
        return self.to_pymilo_issue()
