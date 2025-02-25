# -*- coding: utf-8 -*-
"""PyMilo Feature Extraction transporter."""
from scipy.sparse import csr_matrix

from ..pymilo_param import SKLEARN_FEATURE_EXTRACTION_TABLE
from ..utils.util import check_str_in_iterable, get_sklearn_type
from .transporter import AbstractTransporter, Command
from .general_data_structure_transporter import GeneralDataStructureTransporter
from .randomstate_transporter import RandomStateTransporter

FEATURE_EXTRACTION_CHAIN = {
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
}


class FeatureExtractorTransporter(AbstractTransporter):
    """Feature Extractor object dedicated Transporter."""

    def serialize(self, data, key, model_type):
        """
        Serialize Feature Extractor object.

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
        if self.is_fe_module(data[key]):
            return self.serialize_fe_module(data[key])
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized feature extraction object.

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
        if self.is_fe_module(content):
            return self.deserialize_fe_module(content)
        return content

    def is_fe_module(self, fe_module):
        """
        Check whether the given module is a sklearn Feature Extraction module or not.

        :param fe_module: given object
        :type fe_module: any
        :return: bool
        """
        if isinstance(fe_module, dict):
            return check_str_in_iterable(
                "pymilo-feature_extraction-type",
                fe_module) and fe_module["pymilo-feature_extraction-type"] in SKLEARN_FEATURE_EXTRACTION_TABLE
        return get_sklearn_type(fe_module) in SKLEARN_FEATURE_EXTRACTION_TABLE

    def serialize_fe_module(self, fe_module):
        """
        Serialize Feature Extraction object.

        :param fe_module: given sklearn feature extraction module
        :type fe_module: sklearn.feature_extraction
        :return: pymilo serialized fe_module
        """
        # add one depth inner preprocessing module population
        for key, value in fe_module.__dict__.items():
            if self.is_fe_module(value):
                fe_module.__dict__[key] = self.serialize_fe_module(value)
            elif isinstance(value, csr_matrix):
                fe_module.__dict__[key] = {
                    "pymilo-bypass": True,
                    "pymilo-csr_matrix": FEATURE_EXTRACTION_CHAIN["GeneralDataStructureTransporter"].serialize_dict(
                        value.__dict__
                    )
                }

        for transporter in FEATURE_EXTRACTION_CHAIN:
            FEATURE_EXTRACTION_CHAIN[transporter].transport(
                fe_module, Command.SERIALIZE)
        return {
            "pymilo-bypass": True,
            "pymilo-feature_extraction-type": get_sklearn_type(fe_module),
            "pymilo-feature_extraction-data": fe_module.__dict__
        }

    def deserialize_fe_module(self, serialized_fe_module):
        """
        Deserialize Feature Extraction object.

        :param serialized_fe_module: serializezd feature extraction module(by pymilo)
        :type serialized_fe_module: dict
        :return: retrieved associated sklearn.feature_extraction module
        """
        data = serialized_fe_module["pymilo-feature_extraction-data"]
        associated_type = SKLEARN_FEATURE_EXTRACTION_TABLE[serialized_fe_module["pymilo-feature_extraction-type"]]
        retrieved_fe_module = associated_type()
        for key in data:
            # add one depth inner feature extraction module population
            if self.is_fe_module(data[key]):
                data[key] = self.deserialize_fe_module(data[key])
            elif check_str_in_iterable("pymilo-csr_matrix", data[key]):
                csr_matrix_dict = FEATURE_EXTRACTION_CHAIN["GeneralDataStructureTransporter"].get_deserialized_dict(
                    data[key]["pymilo-csr_matrix"])
                cm = csr_matrix(csr_matrix_dict['_shape'])
                for _key in csr_matrix_dict:
                    setattr(cm, _key, csr_matrix_dict[_key])
                data[key] = cm
            for transporter in FEATURE_EXTRACTION_CHAIN:
                data[key] = FEATURE_EXTRACTION_CHAIN[transporter].deserialize(data, key, "")
        for key in data:
            setattr(retrieved_fe_module, key, data[key])
        return retrieved_fe_module
