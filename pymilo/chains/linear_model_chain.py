# -*- coding: utf-8 -*-
"""PyMilo chain for linear models."""

from .chain import AbstractChain
from ..transporters.baseloss_transporter import BaseLossTransporter
from ..transporters.transporter import Command
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.lossfunction_transporter import LossFunctionTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..utils.util import get_sklearn_type, is_iterable
from ..pymilo_param import SKLEARN_LINEAR_MODEL_TABLE

LINEAR_MODEL_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "BaseLossTransporter": BaseLossTransporter(),
    "LossFunctionTransporter": LossFunctionTransporter(),
}


class LinearModelChain(AbstractChain):
    """LinearModelChain developed to handle sklearn Linear ML model transportation."""

    def serialize(self, linear_model_object):
        """
        Return the serialized json string of the given linear model.

        :param linear_model_object: given model to be get serialized
        :type linear_model_object: any sklearn linear model
        :return: the serialized json string of the given linear model
        """
        # first serializing the inner linear models...
        for key in linear_model_object.__dict__:
            if self.is_supported(linear_model_object.__dict__[key]):
                linear_model_object.__dict__[key] = {
                    "pymilo-inner-model-data": self.transport(linear_model_object.__dict__[key], Command.SERIALIZE, True),
                    "pymilo-inner-model-type": get_sklearn_type(linear_model_object.__dict__[key]),
                    "pymilo-bypass": True
                }
        # now serializing non-linear model fields
        for transporter in self._transporters:
            self._transporters[transporter].transport(
                linear_model_object, Command.SERIALIZE)
        return linear_model_object.__dict__

    def deserialize(self, linear_model, is_inner_model=False):
        """
        Return the associated sklearn linear model of the given linear_model.

        :param linear_model: given json string of a linear model to get deserialized to associated sklearn linear model
        :type linear_model: obj
        :param is_inner_model: determines whether it is an inner model of a super ml model
        :type is_inner_model: boolean
        :return: associated sklearn linear model
        """
        raw_model = None
        data = None
        if is_inner_model:
            raw_model = self._supported_models[linear_model["type"]]()
            data = linear_model["data"]
        else:
            raw_model = self._supported_models[linear_model.type]()
            data = linear_model.data
        # first deserializing the inner linear models(one depth inner linear
        # models have been deserialized -> TODO full depth).
        for key in data:
            if is_deserialized_linear_model(data[key]):
                data[key] = self.transport({
                    "data": data[key]["pymilo-inner-model-data"],
                    "type": data[key]["pymilo-inner-model-type"]
                }, Command.DESERIALIZE, is_inner_model=True)
        # now deserializing non-linear models fields
        for transporter in self._transporters:
            self._transporters[transporter].transport(
                linear_model, Command.DESERIALIZE, is_inner_model)
        for item in data:
            setattr(raw_model, item, data[item])
        return raw_model


linear_chain = LinearModelChain(LINEAR_MODEL_CHAIN, SKLEARN_LINEAR_MODEL_TABLE)


def is_deserialized_linear_model(content):
    """
    Check if the given content is a previously serialized model by Pymilo's Export or not.

    :param content: given object to be authorized as a valid pymilo exported serialized model
    :type content: any object
    :return: check result as bool
    """
    if not is_iterable(content):
        return False
    return "pymilo-inner-model-type" in content and "pymilo-inner-model-data" in content
