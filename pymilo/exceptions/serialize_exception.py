from enum import Enum

class SerilaizatoinErrorTypes(Enum):
    INVALID_MODEL = 1
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 2
    UNKNOWN = 3

class PymiloSerializationException(Exception):
    def __init__(self, meta_data):            
        # Call the base class constructor with the parameters it needs
        message = "TODO: Comprehensive error message for serialization."
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        # TODO
        return 