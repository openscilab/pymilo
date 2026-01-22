# -*- coding: utf-8 -*-
"""PyMilo Compose transporter."""

from ..utils.util import check_str_in_iterable
from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter
from .preprocessing_transporter import PreprocessingTransporter
from .feature_extraction_transporter import FeatureExtractorTransporter
from ..chains.util import serialize_possible_ml_model, deserialize_possible_ml_model

COMPOSE_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "PreprocessingTransporter": PreprocessingTransporter(),
    "FeatureExtractorTransporter": FeatureExtractorTransporter(),
}


class ComposeTransporter(AbstractTransporter):
    """Compose object dedicated Transporter."""

    def is_compose_internal_model(self, internal_model):
        """
        Check whether the given content is a nested estimator/module.

        This is used to detect objects that should be (de)serialized when found inside
        compose models (e.g., inside ColumnTransformer.transformers).

        :param internal_model: given object
        :type internal_model: any
        :return: bool
        """
        pt = COMPOSE_CHAIN["PreprocessingTransporter"]
        fe = COMPOSE_CHAIN["FeatureExtractorTransporter"]
        if isinstance(internal_model, dict):
            return (
                check_str_in_iterable("pymilo-inner-model-type", internal_model) or
                pt.is_preprocessing_module(internal_model) or
                fe.is_fe_module(internal_model)
            )
        return (
            self._is_ml_model(internal_model) or
            pt.is_preprocessing_module(internal_model) or
            fe.is_fe_module(internal_model)
        )

    def _is_ml_model(self, obj):
        """
        Check if the object is an ML model that needs serialization.

        :param obj: given object
        :type obj: any
        :return: bool
        """
        return hasattr(obj, 'fit') and (hasattr(obj, 'predict') or hasattr(obj, 'transform'))

    def serialize_compose_internal_model(self, internal_model):
        """
        Serialize internal model of compose objects.

        :param internal_model: given sklearn internal model
        :type internal_model: sklearn model or function
        :return: pymilo serialized internal_model
        """
        # Handle preprocessing modules
        pt = COMPOSE_CHAIN["PreprocessingTransporter"]
        if pt.is_preprocessing_module(internal_model):
            return pt.serialize_pre_module(internal_model)

        # Handle feature extraction modules
        fe = COMPOSE_CHAIN["FeatureExtractorTransporter"]
        if fe.is_fe_module(internal_model):
            return fe.serialize_fe_module(internal_model)

        # Handle ML models (including compose/ensemble/concrete estimators) using the project-wide schema.
        if self._is_ml_model(internal_model):
            has_ml_model, result = serialize_possible_ml_model(internal_model)
            if has_ml_model:
                return result
        return internal_model

    def deserialize_compose_internal_model(self, serialized_internal_model):
        """
        Deserialize internal model of compose objects.

        :param serialized_internal_model: serialized internal model(by pymilo)
        :type serialized_internal_model: dict
        :return: retrieved associated sklearn internal model
        """
        # Preprocessing / feature extraction modules
        pt = COMPOSE_CHAIN["PreprocessingTransporter"]
        if pt.is_preprocessing_module(serialized_internal_model):
            return pt.deserialize_pre_module(serialized_internal_model)

        fe = COMPOSE_CHAIN["FeatureExtractorTransporter"]
        if fe.is_fe_module(serialized_internal_model):
            return fe.deserialize_fe_module(serialized_internal_model)

        # Project-wide nested-model schema
        has_ml_model, result = deserialize_possible_ml_model(serialized_internal_model)
        if has_ml_model:
            return result

        return serialized_internal_model

    def _serialize_nested(self, obj):
        """
        Recursively serialize nested structures containing internal models.

        :param obj: object to serialize (dict, list, tuple, or internal model)
        :type obj: any
        :return: serialized object with internal models converted to pymilo format
        """
        if isinstance(obj, dict):
            return {k: self._serialize_nested(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._serialize_nested(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._serialize_nested(v) for v in obj)
        if self.is_compose_internal_model(obj):
            return self.serialize_compose_internal_model(obj)
        return obj

    def _deserialize_nested(self, obj):
        """
        Recursively deserialize nested structures containing serialized internal models.

        :param obj: object to deserialize (dict, list, tuple, or serialized internal model)
        :type obj: any
        :return: deserialized object with internal models restored to sklearn objects
        """
        if isinstance(obj, dict):
            # Try leaf deserialization first (nested model schemas are dicts)
            if self.is_compose_internal_model(obj):
                return self.deserialize_compose_internal_model(obj)
            return {k: self._deserialize_nested(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._deserialize_nested(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._deserialize_nested(v) for v in obj)
        return obj

    def serialize(self, data, key, model_type):
        """
        Serialize Compose object.

        Serialize the data[key] of the given model which type is model_type.
        To fully serialize a model, we should traverse over all the keys of its data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which data dictionary is given as the data param
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], (dict, list, tuple)):
            return self._serialize_nested(data[key])
        if self.is_compose_internal_model(data[key]):
            return self.serialize_compose_internal_model(data[key])
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized compose object.

        Deserialize the data[key] of the given model which type is model_type.
        To fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by
        pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which internal serialized data dictionary is given as the data param
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        if isinstance(data[key], (dict, list, tuple)):
            return self._deserialize_nested(data[key])
        if self.is_compose_internal_model(data[key]):
            return self.deserialize_compose_internal_model(data[key])
        return data[key]
