import pymilo
import sklearn
import platform
from datetime import datetime
from abc import ABC, abstractmethod


class PymiloException(Exception, ABC):
    """
    PymiloException is an abstract class for handling pymilo associated exceptions.
    """
    def __init__(self, message, meta_data):
        """
        initialize the PymiloException instance.

        :param meta_data: Details pertain to the populated error.
        :type meta_data: dictionary[str:str]
        :param message: Error message associated with the populated error.
        :type message: str
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

        :return: a dictionary of error's details.
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

        :return: issue form of the associated error as a string
        """
        pass

    def __str__(self):
        return self.to_pymilo_issue()
