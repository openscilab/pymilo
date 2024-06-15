# -*- coding: utf-8 -*-
"""PyMilo BinMapper transporter."""
from sklearn.ensemble._hist_gradient_boosting.binning import _BinMapper
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter


class BinMapperTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle _BinMapper objects."""

    def serialize(self, data, key, model_type):
        """
        Serialize _BinMapper object[useful in HistGradientBoosting(Regressor,Classifier)].

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
        if isinstance(data[key], _BinMapper):
            binMapper = data[key]
            _data = binMapper.__dict__
            gdst = GeneralDataStructureTransporter()
            for _key in _data:
                _data[_key] = gdst.serialize(_data, _key, model_type + ":_BinMapper")
            return {
                "pymilo-bypass": True,
                "pymilo-binmapper": {
                    "__dict__": _data
                }
            }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized _BinMapper object[useful in HistGradientBoosting(Regressor,Classifier)].

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
        if is_primitive(content) or content is None:
            return content

        if check_str_in_iterable("pymilo-binmapper", content):
            __dict__ = content["pymilo-binmapper"]["__dict__"]
            binMapper = _BinMapper()
            gdst = GeneralDataStructureTransporter()
            for key in __dict__:
                __dict__[key] = gdst.deserialize(__dict__, key, model_type + ":_BinMapper")
            for key in __dict__:
                setattr(binMapper, key, __dict__[key])
            return binMapper

        return content
