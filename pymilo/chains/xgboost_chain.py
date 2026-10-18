# -*- coding: utf-8 -*-
"""PyMilo chain for XGBoost models."""

from .chain import AbstractChain
from ..transporters.transporter import Command
from ..transporters.booster_transporter import BoosterTransporter
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter

from ..pymilo_param import XGBOOST_SUPPORTED_TABLE, NOT_SUPPORTED
from ..utils.util import get_sklearn_type

_xgb = None
_xgboost_available = False
try:
    import xgboost as _xgb_imported
    _xgb = _xgb_imported
    _xgboost_available = True
except BaseException:
    _xgb = None
    _xgboost_available = False


XGBOOST_CHAIN = {
    "BoosterTransporter": BoosterTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
}


class XGBoostChain(AbstractChain):
    """XGBoostChain handles export/import of XGBoost ML model objects.

    Two model shapes are supported and routed via isinstance checks (no hardcoded keys):

    1. The scikit-learn-style wrappers (XGBRegressor, XGBClassifier, XGBRanker,
       XGBRFRegressor, XGBRFClassifier). These have a regular Python __dict__ that is
       walked through the transporter chain. The inner xgboost.Booster (typically stored
       under the '_Booster' attribute) is handled by BoosterTransporter via
       isinstance(value, xgboost.Booster); everything else (ndarrays, classes_,
       evals_result_, scalar hyperparameters, etc.) is handled by
       GeneralDataStructureTransporter.

    2. The native xgboost.Booster (returned by xgboost.train). It has no usable
       Python __dict__, so it is wrapped in a synthetic single-key dict and serialized
       directly via BoosterTransporter.
    """

    def is_supported(self, model):
        """
        Check if the given model is an XGBoost ML model supported by this chain.

        :param model: a string name of an ML model or an XGBoost object of it
        :type model: any object
        :return: check result as bool
        """
        if not _xgboost_available:
            return False

        if isinstance(model, str):
            return model in self._supported_models and self._supported_models[model] != NOT_SUPPORTED

        if isinstance(model, _xgb.Booster):
            return self._supported_models.get("Booster", NOT_SUPPORTED) != NOT_SUPPORTED

        try:
            from xgboost.sklearn import XGBModel
        except BaseException:
            XGBModel = None

        if XGBModel is not None and isinstance(model, XGBModel):
            model_name = get_sklearn_type(model)
            return model_name in self._supported_models and self._supported_models[model_name] != NOT_SUPPORTED

        return False

    def serialize(self, model):
        """
        Return the serialized data dict of the given XGBoost model.

        :param model: given XGBoost model to be serialized
        :type model: xgboost sklearn-style estimator or xgboost.Booster
        :return: dict
        """
        if _xgboost_available and isinstance(model, _xgb.Booster):
            shell = {"_booster": model}
            shell["_booster"] = self._transporters["BoosterTransporter"].serialize(
                shell, "_booster", "Booster")
            return shell

        for transporter in self._transporters:
            self._transporters[transporter].transport(model, Command.SERIALIZE)
        return model.__dict__

    def deserialize(self, serialized_model, is_inner_model=False):
        """
        Return the associated XGBoost model of the previously serialized model.

        :param serialized_model: pymilo Import object or inner-model dict
        :type serialized_model: obj
        :param is_inner_model: whether this is being deserialized as a nested model
        :type is_inner_model: boolean
        :return: associated XGBoost model
        """
        if is_inner_model:
            data = serialized_model["data"]
            model_type = serialized_model["type"]
        else:
            data = serialized_model.data
            model_type = serialized_model.type

        if model_type == "Booster":
            booster_value = data.get("_booster", None)
            return self._transporters["BoosterTransporter"].deserialize(
                {"_booster": booster_value}, "_booster", model_type)

        raw_model = self._supported_models[model_type]()

        for transporter in self._transporters:
            self._transporters[transporter].transport(
                serialized_model, Command.DESERIALIZE, is_inner_model)

        for item in data:
            setattr(raw_model, item, data[item])
        return raw_model


xgboost_chain = XGBoostChain(XGBOOST_CHAIN, XGBOOST_SUPPORTED_TABLE)
