# -*- coding: utf-8 -*-
"""PyMilo chain for compose models."""

from ..chains.chain import AbstractChain
from ..transporters.compose_transporter import ComposeTransporter
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.function_transporter import FunctionTransporter
from ..transporters.transporter import Command
from ..pymilo_param import SKLEARN_COMPOSE_TABLE

COMPOSE_CHAIN = {
    "ComposeTransporter": ComposeTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "FunctionTransporter": FunctionTransporter(),
}


class ComposeModelChain(AbstractChain):
    """ComposeModelChain developed to handle sklearn Compose ML model transportation."""

    def deserialize(self, compose, is_inner_model=False):
        """
        Return the associated sklearn compose model of the given compose.

        :param compose: given json string of a compose model to get deserialized to associated sklearn compose model
        :type compose: obj
        :param is_inner_model: determines whether it is an inner compose model of a super ml model
        :type is_inner_model: boolean
        :return: associated sklearn compose model
        """
        data = compose["data"] if is_inner_model else compose.data
        _type = compose["type"] if is_inner_model else compose.type

        # ColumnTransformer requires 'transformers' arg; others use default constructor
        if _type == "ColumnTransformer":
            raw_model = self._supported_models[_type](transformers=data.get("transformers", []))
        else:
            raw_model = self._supported_models[_type]()

        for transporter in self._transporters:
            self._transporters[transporter].transport(compose, Command.DESERIALIZE, is_inner_model)

        for item in data:
            setattr(raw_model, item, data[item])
        return raw_model


compose_chain = ComposeModelChain(COMPOSE_CHAIN, SKLEARN_COMPOSE_TABLE)


def get_transporter(model):
    """
    Get associated transporter for the given ML model.

    :param model: given model to get it's transporter
    :type model: scikit ML model
    :return: model category and transporter function
    """
    if isinstance(model, str):
        if model.upper() == "COMPOSE":
            return "COMPOSE", compose_chain.transport
    if compose_chain.is_supported(model):
        return "COMPOSE", compose_chain.transport
    else:
        return None, None
