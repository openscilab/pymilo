# -*- coding: utf-8 -*-
"""PyMilo LabelEncoder transporter."""
from sklearn.preprocessing import LabelEncoder
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter


class LabelEncoderTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to LabelEncoder objects."""

    def serialize(self, data, key, model_type):
        """
        Serialize LabelEncoder object.

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
        if isinstance(data[key], LabelEncoder):
            label_encoder = data[key]
            data[key] = {
                "pymilo-bypass": True, "pymilo-labelencoder": {
                    "classes_": GeneralDataStructureTransporter().deep_serialize_ndarray(
                        label_encoder.__dict__["classes_"])}}
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized LabelEncoder object.

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
        if is_primitive(content) or isinstance(content, type(None)):
            return content

        if check_str_in_iterable("pymilo-labelencoder", content):
            serialized_le = content["pymilo-labelencoder"]
            label_encoder = LabelEncoder()
            setattr(
                label_encoder,
                "classes_",
                GeneralDataStructureTransporter().deep_deserialize_ndarray(
                    serialized_le["classes_"]))
            return label_encoder

        return content
