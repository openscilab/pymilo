from ..transporters.transporter import Command

from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.baseloss_transporter import BaseLossTransporter
from ..transporters.lossfunction_transporter import LossFunctionTransporter
from ..transporters.labelbinarizer_transporter import LabelBinarizerTransporter

from ..pymilo_param import SKLEARN_MODEL_TABLE
from ..utils.util import get_sklearn_type, is_iterable

LINEAR_MODEL_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "BaseLossTransporter": BaseLossTransporter(),
    "LossFunctionTransporter": LossFunctionTransporter(),
    "LabelBinarizerTransporter": LabelBinarizerTransporter()}


def is_linear_model(model):
    return type(model) in SKLEARN_MODEL_TABLE.values()


def is_deserialized_linear_model(content):
    if not (is_iterable(content)):
        return False
    return "inner-model-type" in content and "inner-model-data" in content


def transport_linear_model(request, command, is_inner_model=False):

    if (command == Command.SERIALIZE):
        # first serializing the inner linear models...
        for key in request.__dict__.keys():
            if is_linear_model(request.__dict__[key]):
                request.__dict__[key] = {
                    "inner-model-data": transport_linear_model(request.__dict__[key], Command.SERIALIZE),
                    "inner-model-type": get_sklearn_type(request.__dict__[key]),
                    "by-pass": True
                }
        # now serializing non-linear model fields
        for transporter in LINEAR_MODEL_CHAIN.keys():
            LINEAR_MODEL_CHAIN[transporter].transport(
                request, Command.SERIALIZE)
        return request.__dict__

    elif (command == Command.DESERIALZIE):
        raw_model = None
        data = None
        if (is_inner_model):
            raw_model = SKLEARN_MODEL_TABLE[request["type"]]()
            data = request["data"]
        else:
            raw_model = SKLEARN_MODEL_TABLE[request.type]()
            data = request.data
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
                request, Command.DESERIALZIE, is_inner_model)
        for item in data.keys():
            setattr(raw_model, item, data[item])
        return raw_model
    else:
        return None  # TODO error handling.
