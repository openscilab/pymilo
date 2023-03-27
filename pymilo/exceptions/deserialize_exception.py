from enum import Enum

class DeSerilaizatoinErrorTypes(Enum):
    CORRUPTED_JSON_FILE = 1
    INVALID_MODEL = 2
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3
    UNKNOWN = 4

class PymiloDeserializationException(Exception):
    def __init__(self, meta_data):            
        # Call the base class constructor with the parameters it needs
        message = "TODO: Comprehensive error message for deserialization."
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        # TODO
        return 