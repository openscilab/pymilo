# -*- coding: utf-8 -*-
"""PyMilo XGBoost Booster transporter."""
import json

from .transporter import AbstractTransporter
from ..utils.util import check_str_in_iterable

_xgb = None
_xgboost_available = False
try:
    import xgboost as _xgb_imported
    _xgb = _xgb_imported
    _xgboost_available = True
except BaseException:
    _xgb = None
    _xgboost_available = False


class BoosterTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle xgboost.Booster fields.

    The Booster's model state lives behind a C handle and cannot be reconstructed by walking
    its Python __dict__. Therefore this transporter routes via isinstance(value, xgboost.Booster)
    only (no key-name match) and relies on XGBoost's officially documented, version-stable JSON
    model format. The booster JSON is parsed and inlined as a regular nested dict so the
    resulting PyMilo JSON is fully transparent: every tree, leaf, and split condition stays
    human-readable and editable.
    """

    PYMILO_BOOSTER_KEY = "pymilo-xgb-booster"

    def serialize(self, data, key, model_type):
        """
        Serialize the value at data[key] if it is an xgboost.Booster instance.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        value = data[key]
        if not _xgboost_available:
            return value
        if not isinstance(value, _xgb.Booster):
            return value
        raw = value.save_raw(raw_format='json')
        if isinstance(raw, (bytes, bytearray)):
            raw = bytes(raw).decode('utf-8')
        booster_dict = json.loads(raw)
        return {
            self.PYMILO_BOOSTER_KEY: True,
            "pymilo-bypass": True,
            "data": booster_dict,
        }

    def deserialize(self, data, key, model_type):
        """
        Deserialize a previously serialized xgboost.Booster value.

        :param data: the internal data dictionary of the associated json file of the ML model
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if not (isinstance(content, dict) and check_str_in_iterable(self.PYMILO_BOOSTER_KEY, content)):
            return content
        if not _xgboost_available:
            return content
        booster_payload = json.dumps(content["data"]).encode('utf-8')
        booster = _xgb.Booster()
        booster.load_model(bytearray(booster_payload))
        return booster
