from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.baseloss_transporter import BaseLossTransporter
from ..transporters.lossfunction_transporter import LossFunctionTransporter
from ..transporters.labelbinarizer_transporter import LabelBinarizerTransporter

from ..pymilo_param import SKLEARN_LINEAR_MODEL_TABLE
from ..utils.util import get_sklearn_type, is_iterable

from ..exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from traceback import format_exc


LINEAR_MODEL_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "BaseLossTransporter": BaseLossTransporter(),
    "LossFunctionTransporter": LossFunctionTransporter(),
    "LabelBinarizerTransporter": LabelBinarizerTransporter()}


def is_linear_model(model):
    return type(model) in SKLEARN_LINEAR_MODEL_TABLE.values()


def is_deserialized_linear_model(content):
    if not is_iterable(content):
        return False
    return "inner-model-type" in content and "inner-model-data" in content


def transport_linear_model(request, command, is_inner_model=False):

    validate_input(request, command)

    if (command == Command.SERIALIZE):
        try:
            return serialize_linear_model(request)
        except Exception as e:
            raise PymiloSerializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()
                    },
                    'object': request
                }
            )
        
    elif command == Command.DESERIALZIE:
        return deserialize_linear_model(request, is_inner_model)

def serialize_linear_model(linear_model_object):
    # first serializing the inner linear models...
    for key in linear_model_object.__dict__.keys():
        if is_linear_model(linear_model_object.__dict__[key]):
            linear_model_object.__dict__[key] = {
                "inner-model-data": transport_linear_model(linear_model_object.__dict__[key], Command.SERIALIZE),
                "inner-model-type": get_sklearn_type(linear_model_object.__dict__[key]),
                "by-pass": True
            }
    # now serializing non-linear model fields
    for transporter in LINEAR_MODEL_CHAIN.keys():
        LINEAR_MODEL_CHAIN[transporter].transport(
            linear_model_object, Command.SERIALIZE)
    return linear_model_object.__dict__


def deserialize_linear_model(linear_model_json, is_inner_model):
    raw_model = None
    data = None
    if (is_inner_model):
        raw_model = SKLEARN_LINEAR_MODEL_TABLE[linear_model_json["type"]]()
        data = linear_model_json["data"]
    else:
        raw_model = SKLEARN_LINEAR_MODEL_TABLE[linear_model_json.type]()
        data = linear_model_json.data
    # first deserializing the inner linear models(one depth inner linear
    # models have been deserialized -> TODO full depth).
    for key in data.keys():
        if is_deserialized_linear_model(data[key]):
            data[key] = transport_linear_model({
                "data": data[key]["inner-model-data"],
                "type": data[key]["inner-model-type"]
            }, Command.DESERIALZIE, is_inner_model=True)
    # now deserializing non-linear models fields
    for transporter in LINEAR_MODEL_CHAIN.keys():
        LINEAR_MODEL_CHAIN[transporter].transport(
            linear_model_json, Command.DESERIALZIE, is_inner_model)
    for item in data.keys():
        setattr(raw_model, item, data[item])
    return raw_model


def validate_input(model, command):
    if(command == Command.SERIALIZE):
        if(type(model) in SKLEARN_LINEAR_MODEL_TABLE.keys()):
            return 
        else: 
            raise PymiloSerializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif(command == Command.DESERIALZIE):
        return "TODO"
    
