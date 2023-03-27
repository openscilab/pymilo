from enum import Enum
import pymilo
import sklearn
import platform
from datetime import datetime
class SerilaizatoinErrorTypes(Enum):
    INVALID_MODEL = 1
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 2
    UNKNOWN = 3

class PymiloSerializationException(Exception):
    def __init__(self, meta_data):            
        # Call the base class constructor with the parameters it needs
        message = "Pymilo Serialization failed since "
        error_type = meta_data['error_type']
        if(error_type == SerilaizatoinErrorTypes.INVALID_MODEL):
            message += 'the given model is not supported or is not a valid model.'
        elif(error_type == SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE):
            message += 'the given model has some non-standard customized internal objects or functions.'
        elif(error_type == SerilaizatoinErrorTypes.UNKNOWN):
            message =  'an unknown error populated(please report).'
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        pymilo_report = super().to_pymilo_log()
        # TODO add any serializable field to `object` field of pymilo_report
        return pymilo_report