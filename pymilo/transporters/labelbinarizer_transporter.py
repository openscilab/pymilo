from ..pymilo_param import KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION
from sklearn import preprocessing
import numpy as np
from .transporter import AbstractTransporter

# TODO exception handling
# Handling Label Binarizer for Ridge Classifier(+[CV])


class LabelBinarizerTransporter(AbstractTransporter):

    # SERIALIZATION
    def serialize(self, data, key, model_type):
        """
        serialize the LabelBinarizer field(if there is).
        """
        if isinstance(data[key], preprocessing.LabelBinarizer):
            data[key] = self.get_serialized_label_binarizer(data[key])
        return data[key]

    def get_serialized_label_binarizer(self, label_binarizer):
        """
        serialize a LabelBinarizer object.
        :param label_binarizer: a label_binarizer object
        :type label_binarizer: sklearn.preprocessing.LabelBinarizer
        :return: pymilo serialized output of label_binarizer object
        """
        data = label_binarizer.__dict__
        for key in data.keys():
            if isinstance(data[key], np.ndarray):
                data[key] = data[key].tolist()
        return data

    # DESERIALIZATION
    def deserialize(self, data, key, model_type):
        """
        deserialize the LabelBinarizer field(if there is).
        """
        content = data[key]
        if key != "_label_binarizer":
            return content
        return self.get_deserialized_label_binarizer(content)

    def get_deserialized_label_binarizer(self, content):
        """
        deserialize the pymilo serialized labelBinarizer field of the associated ML model.
        :param content: a label_binarizer object
        :type content: sklearn.preprocessing.LabelBinarizer
        :return: a sklearn.preprocessing.LabelBinarizer instance derived from the 
        pymilo deserialized output of the previously pymilo serialized label_binarizer field.
        """
        raw_lb = KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION["_label_binarizer"](
        )
        for key in content.keys():
            if isinstance(content[key], list):
                content[key] = np.array(content[key])
        for item in content.keys():
            setattr(raw_lb, item, content[item])
        return raw_lb
