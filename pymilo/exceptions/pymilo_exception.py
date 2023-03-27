from abc import ABC, abstractmethod

class PymiloException(Exception, ABC):
    def __init__(self, message, meta_data):            
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        # gathered meta_data
        self.meta_data = meta_data

    @abstractmethod
    def to_pymilo_log(self):
        pass