import pymilo
import sklearn
import platform
from datetime import datetime
from abc import ABC, abstractmethod


class PymiloException(Exception, ABC):
    def __init__(self, message, meta_data):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # gathered meta_data
        self.message = message
        self.meta_data = meta_data

    # collect All pymilo related data.
    def to_pymilo_log(self):
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
        pass

    def __str__(self):
        return self.to_pymilo_issue()
