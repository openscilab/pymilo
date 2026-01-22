# -*- coding: utf-8 -*-
"""PyMilo Preprocessing transporter."""
from ..pymilo_param import SKLEARN_PREPROCESSING_TABLE
from ..utils.util import check_str_in_iterable, get_sklearn_type
from .transporter import AbstractTransporter, Command
from .general_data_structure_transporter import GeneralDataStructureTransporter
from .function_transporter import FunctionTransporter
from scipy.interpolate._bsplines import BSpline

PREPROCESSING_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "FunctionTransporter": FunctionTransporter(),
}


class PreprocessingTransporter(AbstractTransporter):
    """Preprocessing object dedicated Transporter."""

    def serialize(self, data, key, model_type):
        """
        Serialize Preprocessing object.

        serialize the data[key] of the given model which type is model_type.
        basically in order to fully serialize a model, we should traverse over all the keys of its data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which data dictionary is given as the data param
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if self.is_preprocessing_module(data[key]):
            return self.serialize_pre_module(data[key])
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized preprocessing object.

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
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
        content = data[key]
        if self.is_preprocessing_module(content):
            return self.deserialize_pre_module(content)
        return content

    def is_preprocessing_module(self, pre_module):
        """
        Check whether the given module is a sklearn Preprocessing module or not.

        :param pre_module: given object
        :type pre_module: any
        :return: bool
        """
        if isinstance(pre_module, dict):
            return check_str_in_iterable(
                "pymilo-preprocessing-type",
                pre_module) and pre_module["pymilo-preprocessing-type"] in SKLEARN_PREPROCESSING_TABLE.keys()
        return get_sklearn_type(pre_module) in SKLEARN_PREPROCESSING_TABLE.keys()

    def serialize_pre_module(self, pre_module):
        """
        Serialize Preprocessing object.

        :param pre_module: given sklearn preprocessing module
        :type pre_module: sklearn.preprocessing
        :return: pymilo serialized pre_module
        """
        # add one depth inner preprocessing module population
        for key, value in pre_module.__dict__.items():
            if self.is_preprocessing_module(value):
                pre_module.__dict__[key] = self.serialize_pre_module(value)
            elif isinstance(value, BSpline):
                pre_module.__dict__[key] = self.serialize_spline(value)
            elif isinstance(value, list):
                if len(value) > 0 and isinstance(value[0], BSpline):
                    pre_module.__dict__[key] = [self.serialize_spline(bspline) for bspline in value]

        for transporter in PREPROCESSING_CHAIN:
            PREPROCESSING_CHAIN[transporter].transport(
                pre_module, Command.SERIALIZE)
        return {
            "pymilo-bypass": True,
            "pymilo-preprocessing-type": get_sklearn_type(pre_module),
            "pymilo-preprocessing-data": pre_module.__dict__
        }

    def deserialize_pre_module(self, serialized_pre_module):
        """
        Deserialize Preprocessing object.

        :param serialized_pre_module: serializezd preprocessing module(by pymilo)
        :type serialized_pre_module: dict
        :return: retrieved associated sklearn.preprocessing module
        """
        data = serialized_pre_module["pymilo-preprocessing-data"]
        associated_type = SKLEARN_PREPROCESSING_TABLE[serialized_pre_module["pymilo-preprocessing-type"]]
        retrieved_pre_module = associated_type()
        for key in data:
            # add one depth inner preprocessing module population
            if self.is_preprocessing_module(data[key]):
                data[key] = self.deserialize_pre_module(data[key])
            # check inner field is BSpline
            if self.is_bspline(data[key]):
                data[key] = self.deserialize_spline(data[key])
            # check inner field is [BSpline]
            if isinstance(data[key], list) and len(data[key]) > 0 and self.is_bspline(data[key][0]):
                data[key] = [self.deserialize_spline(serialized_bspline) for serialized_bspline in data[key]]

            for transporter in PREPROCESSING_CHAIN:
                data[key] = PREPROCESSING_CHAIN[transporter].deserialize(data, key, "")
        for key in data:
            setattr(retrieved_pre_module, key, data[key])
        return retrieved_pre_module

    def is_bspline(self, bspline):
        """
        Check whether the given module is a scipy.interpolate._bsplines.BSpline or not.

        :param bspline: given object
        :type bspline: any
        :return: bool
        """
        if isinstance(bspline, dict):
            return check_str_in_iterable(
                "pymilo-preprocessing-type",
                bspline) and bspline["pymilo-preprocessing-type"] == "BSpline"
        return get_sklearn_type(bspline) == "BSpline"

    def serialize_spline(self, bspline):
        """
        Serialize scipy.interpolate._bsplines.BSpline object.

        :param bspline: given scipy.interpolate._bsplines.BSpline module
        :type bspline: scipy.interpolate._bsplines.BSpline
        :return: pymilo serialized bspline
        """
        for transporter in PREPROCESSING_CHAIN:
            PREPROCESSING_CHAIN[transporter].transport(bspline, Command.SERIALIZE)
        return {
            "pymilo-bypass": True,
            "pymilo-preprocessing-type": get_sklearn_type(bspline),
            "pymilo-preprocessing-data": bspline.__dict__
        }

    def deserialize_spline(self, serialized_bspline):
        """
        Deserialize BSpline object.

        :param serialized_bspline: serialized BSpline object(by pymilo)
        :type serialized_bspline: dict
        :return: retrieved associated scipy.interpolate._bsplines.BSpline object
        """
        data = serialized_bspline["pymilo-preprocessing-data"]
        associated_type = BSpline
        for key in list(data.keys()):
            for transporter in PREPROCESSING_CHAIN:
                data[key] = PREPROCESSING_CHAIN[transporter].deserialize(data, key, "")
        # Handle both old scipy (t, c, k) and new scipy (_t, _c, _k) attribute names
        t = data.get("t", data.get("_t"))
        c = data.get("c", data.get("_c"))
        k = data.get("k", data.get("_k"))
        retrieved_pre_module = associated_type(t=t, k=k, c=c)
        for key in data:
            setattr(retrieved_pre_module, key, data[key])
        return retrieved_pre_module
