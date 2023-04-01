from enum import Enum
from .pymilo_exception import PymiloException

class DeSerilaizatoinErrorTypes(Enum):
    CORRUPTED_JSON_FILE = 1
    INVALID_MODEL = 2
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3

class PymiloDeserializationException(PymiloException):
    def __init__(self, meta_data):            
        # Call the base class constructor with the parameters it needs
        message = "Pymilo Deserialization failed since "
        error_type = meta_data['error_type']
        if(error_type == DeSerilaizatoinErrorTypes.CORRUPTED_JSON_FILE):
            message += 'the given json file is not a valid .json file.'
        elif(error_type == DeSerilaizatoinErrorTypes.INVALID_MODEL):
            message += 'the given model is not supported or is not a valid model.'
        elif(error_type == DeSerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE):
            message += 'the given model has some non-standard customized internal objects or functions.'
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        pymilo_report = super().to_pymilo_log()
        if(self.meta_data['error_type'] == DeSerilaizatoinErrorTypes.CORRUPTED_JSON_FILE):
            pymilo_report['object']['json_file'] = self.meta_data['json_file'] 
        return pymilo_report
    
    def to_pymilo_issue(self):
        pymilo_report = self.to_pymilo_log()
        help_request = "\n\nIn order to help us enhance Pymilo's functionality, please open an issue associated with this error and put the message below inside.\n"
        discription = "#### Description\n" + "Pymilo Import failed."
        steps_to_produce = "#### Steps/Code to Reproduce\n" + "It is auto-reported from the pymilo logger."
        expected_behaviour = "#### Expected Behavior\n" + "A successfull Pymilo Import."
        actual_behaviour = "#### Actual Behavior\n" + "Pymilo Import failed."
        operating_system = "#### Operating System\n" + pymilo_report['os']['full-description']
        python_version = "#### Python Version\n" + pymilo_report['versions']["python-version"]
        pymilo_version = "#### PyMilo Version\n" + pymilo_report['versions']["pymilo-version"]
        gathered_data = "#### Logged Data\n" + str(pymilo_report)

        full_issue_form = help_request + discription + steps_to_produce + expected_behaviour + actual_behaviour + operating_system + python_version + pymilo_version + gathered_data
        return full_issue_form