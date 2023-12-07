# -*- coding: utf-8 -*-
"""PyMilo LabelBinarizer transporter."""
from ..pymilo_param import KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION
from sklearn import preprocessing
import numpy as np
from .transporter import AbstractTransporter


class LabelBinarizerTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle LabelBinarizer field(for Ridge Classifier(+[CV]))."""

    def serialize(self, data, key, model_type):
        """
        Serialize the LabelBinarizer field(if there is).

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
        if isinstance(data[key], preprocessing.LabelBinarizer):
            data[key] = self.get_serialized_label_binarizer(data[key])
        return data[key]

    def get_serialized_label_binarizer(self, label_binarizer):
        """
        Serialize a LabelBinarizer object.

        :param label_binarizer: a label_binarizer object
        :type label_binarizer: sklearn.preprocessing.LabelBinarizer
        :return: pymilo serialized output of label_binarizer object
        """
        data = label_binarizer.__dict__
        for key in data:
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        return data

    def deserialize(self, data, key, model_type):
        """
        Deserialize the LabelBinarizer field(if there is).

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file
            of the ML model which is generated previously by pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which internal serialized data dictionary is given as the data param
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if key != "_label_binarizer":
            return content
        return self.get_deserialized_label_binarizer(content)

    def get_deserialized_label_binarizer(self, content):
        """
        Deserialize the pymilo serialized labelBinarizer field of the associated ML model.

        :param content: a label_binarizer object
        :type content: sklearn.preprocessing.LabelBinarizer
        :return: a sklearn.preprocessing.LabelBinarizer instance derived from the
        pymilo deserialized output of the previously pymilo serialized label_binarizer field.
        """
        raw_lb = KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION["_label_binarizer"](
        )
        for item in content:
            setattr(raw_lb, item, content[item])
        return raw_lb
