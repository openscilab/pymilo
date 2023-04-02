from enum import Enum
from .pymilo_exception import PymiloException

class DeSerilaizatoinErrorTypes(Enum):
    CORRUPTED_JSON_FILE = 1
    INVALID_MODEL = 2
    VALID_MODEL_INVALID_INTERNAL_STRUCTURE = 3

class PymiloDeserializationException(PymiloException):
    def __init__(self, meta_data):            
        # Call the base class constructor with the parameters it needs
        message = "Pymilo Deserialization failed since {reason}"
        error_type = meta_data['error_type']
        error_type_to_message = {
            DeSerilaizatoinErrorTypes.CORRUPTED_JSON_FILE: 'the given json file is not a valid .json file.',
            DeSerilaizatoinErrorTypes.INVALID_MODEL: 'the given model is not supported or is not a valid model.',
            DeSerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE: 'the given model has some non-standard customized internal objects or functions.'
        }
        if error_type in error_type_to_message.keys():
            reason = error_type_to_message[error_type]
        else:
            reason = "an Unknown error occurred."
        message.format(reason = reason)
        super().__init__(message, meta_data)

    def to_pymilo_log(self):
        pymilo_report = super().to_pymilo_log()
        if self.meta_data['error_type'] == DeSerilaizatoinErrorTypes.CORRUPTED_JSON_FILE:
            pymilo_report['object']['json_file'] = self.meta_data['json_file'] 
        return pymilo_report
    
    def to_pymilo_issue(self):
        pymilo_report = self.to_pymilo_log()
        help_request = "\n\nIn order to help us enhance Pymilo's functionality, please open an issue associated with this error and put the message below inside.\n"
        discription = "#### Description\n Pymilo Import failed."
        steps_to_produce = "#### Steps/Code to Reproduce\n It is auto-reported from the pymilo logger."
        expected_behaviour = "#### Expected Behavior\n A successfull Pymilo Import."
        actual_behaviour = "#### Actual Behavior\n Pymilo Import failed."
        operating_system = "#### Operating System\n {os}".format(os = pymilo_report['os']['full-description'])
        python_version = "#### Python Version\n {python_version}".format(python_version = pymilo_report['versions']["python-version"])
        pymilo_version = "#### PyMilo Version\n {pymilo_version}".format(pymilo_version = pymilo_report['versions']["pymilo-version"])
        gathered_data = "#### Logged Data\n {logged_data}".format(str(logged_data = pymilo_report))

        full_issue_form = help_request + discription + steps_to_produce + expected_behaviour + actual_behaviour \
        + operating_system + python_version + pymilo_version + gathered_data
        return full_issue_form