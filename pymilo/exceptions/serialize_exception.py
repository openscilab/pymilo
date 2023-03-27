from enum import Enum

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
    
    def to_pymilo_issue(self):
        pymilo_report = self.to_pymilo_log()
        discription = "#### Description\n" + "Pymilo Export failed."
        steps_to_produce = "\n#### Steps/Code to Reproduce\n" + "It is auto-reported from the pymilo logger."
        expected_behaviour = "\n#### Expected Behavior\n" + "A successfull Pymilo Export."
        actual_behaviour = "\n#### Actual Behavior\n" + "Pymilo Export failed."
        operating_system = "\n#### Operating System\n" + pymilo_report['os']['full-description']
        python_version = "\n#### Python Version\n" + pymilo_report['versions']["python-version"]
        pymilo_version = "\n#### PyMilo Version\n" + pymilo_report['versions']["pymilo-version"]
        gathered_data = "\n#### Logged Data\n" + pymilo_report

        full_issue_form = discription + steps_to_produce + expected_behaviour + actual_behaviour + operating_system + python_version + pymilo_version + gathered_data
        return full_issue_form