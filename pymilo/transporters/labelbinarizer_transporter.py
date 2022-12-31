from __future__ import annotations
from ..pymilo_param import KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION
from sklearn import preprocessing
import numpy as np
from .transporter import AbstractTransporter

# TODO exception handling
# Handling Label Binarizer for Ridge Classifier(+[CV])


class LabelBinarizerTransporter(AbstractTransporter):

    # SERIALIZATION
    def serialize(self, data, key, model_type):
        if isinstance(data[key], preprocessing.LabelBinarizer):
            data[key] = self.get_serialized_label_binarizer(data[key])
        return data[key]

    def get_serialized_label_binarizer(self, label_binarizer):
        data = label_binarizer.__dict__
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        return data

    # DESERIALIZATION
    def deserialize(self, data, key, model_type):
        content = data[key]
        if key != "_label_binarizer":
            return content
        return self.get_deserialized_label_binarizer(content)

    def get_deserialized_label_binarizer(self, content):
        raw_lb = KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION["_label_binarizer"](
        )
        for key in content.keys():
            if isinstance(content[key], list):
                content[key] = np.array(content[key])
        for item in content.keys():
            setattr(raw_lb, item, content[item])
        return raw_lb
