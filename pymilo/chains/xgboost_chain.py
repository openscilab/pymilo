# -*- coding: utf-8 -*-
"""PyMilo chain for XGBoost models."""

from ..chains.chain import AbstractChain
from ..pymilo_param import (
    NOT_SUPPORTED,
    XGBOOST_MODEL_TABLE,
    XGBOOST_NOT_INSTALLED,
    xgboost_support,
)
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.transporter import Command
from ..transporters.xgboost_transporter import (
    PYMILO_XGBOOST_BOOSTER,
    XGBoostTransporter,
    booster_from_transparent_dict,
    booster_to_transparent_dict,
    is_xgboost_booster,
    wrap_booster_payload,
)
XGBOOST_CHAIN = {
    "XGBoostTransporter": XGBoostTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
}

PYMILO_XGBOOST_STANDALONE = "pymilo-xgboost-standalone-booster"


class XGBoostModelChain(AbstractChain):
    """XGBoostModelChain developed to handle XGBoost ML model transportation."""

    def is_supported(self, model):
        """
        Check if the given model is an XGBoost model supported by this chain.

        :param model: a string name of an ML model or an XGBoost object of it
        :type model: any object
        :return: check result as bool
        """
        if is_xgboost_booster(model):
            return True
        return super().is_supported(model)

    def serialize(self, model):
        """
        Return the serialized dictionary of the given XGBoost model.

        Standalone Booster objects are exported as a transparent JSON model
        (trees, weights, splits and learner config). Sklearn-compatible wrappers
        keep every constructor parameter plus the same transparent booster body.

        :param model: given model to be get serialized
        :type model: xgboost.core.Booster or xgboost.sklearn.XGBModel
        :return: the serialized dictionary of the given XGBoost model
        """
        if not xgboost_support:
            raise ImportError(XGBOOST_NOT_INSTALLED)
        if is_xgboost_booster(model):
            return {
                PYMILO_XGBOOST_STANDALONE: True,
                PYMILO_XGBOOST_BOOSTER: booster_to_transparent_dict(model),
            }
        for transporter in self._transporters:
            self._transporters[transporter].transport(model, Command.SERIALIZE)
        return model.__dict__

    def deserialize(self, serialized_model, is_inner_model=False):
        """
        Return the associated XGBoost model of the given serialized model.

        :param serialized_model: given json object of an XGBoost model to get deserialized
        :type serialized_model: obj
        :param is_inner_model: determines whether it is an inner model of a super ML model
        :type is_inner_model: boolean
        :return: associated XGBoost model
        """
        if not xgboost_support:
            raise ImportError(XGBOOST_NOT_INSTALLED)

        data = serialized_model["data"] if is_inner_model else serialized_model.data
        model_type = serialized_model["type"] if is_inner_model else serialized_model.type

        if model_type == "Booster" or (
                isinstance(data, dict) and data.get(PYMILO_XGBOOST_STANDALONE)):
            payload = data.get(PYMILO_XGBOOST_BOOSTER, data)
            return booster_from_transparent_dict(payload, map_gpu_to_cpu=True)

        for transporter in self._transporters:
            self._transporters[transporter].transport(
                serialized_model, Command.DESERIALIZE, is_inner_model)

        raw_model = self._instantiate(model_type)
        for item in data:
            try:
                setattr(raw_model, item, data[item])
            except AttributeError:
                # Some XGBoost attributes are read-only properties (e.g. classes_).
                continue
        return raw_model

    def _instantiate(self, model_type):
        """
        Create an empty XGBoost estimator of the requested type.

        :param model_type: model type name
        :type model_type: str
        :return: an unfitted XGBoost estimator
        """
        model_cls = self._supported_models.get(model_type)
        if model_cls is None or model_cls == NOT_SUPPORTED:
            raise ValueError("Unsupported XGBoost model type: {0}".format(model_type))
        return model_cls()


xgboost_chain = XGBoostModelChain(XGBOOST_CHAIN, XGBOOST_MODEL_TABLE)


def is_xgboost_model(model):
    """
    Check whether the given object is a supported XGBoost model.

    :param model: given object or model type name
    :type model: any
    :return: check result as bool
    """
    return xgboost_chain.is_supported(model)


def serialize_xgboost_booster(booster):
    """
    Serialize a standalone XGBoost Booster as a pymilo-bypass payload.

    :param booster: fitted XGBoost Booster
    :type booster: xgboost.core.Booster
    :return: dictionary
    """
    return wrap_booster_payload(booster)


def deserialize_xgboost_booster(payload):
    """
    Deserialize a previously serialized XGBoost Booster payload.

    :param payload: serialized booster dictionary
    :type payload: dict
    :return: xgboost.core.Booster
    """
    if isinstance(payload, dict) and PYMILO_XGBOOST_BOOSTER in payload:
        return booster_from_transparent_dict(payload[PYMILO_XGBOOST_BOOSTER], map_gpu_to_cpu=True)
    return booster_from_transparent_dict(payload, map_gpu_to_cpu=True)
