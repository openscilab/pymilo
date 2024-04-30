# -*- coding: utf-8 -*-
"""PyMilo Preprocessing transporter."""
from ..pymilo_param import SKLEARN_PREPROCESSING_TABLE
from ..utils.util import check_str_in_iterable, get_sklearn_type
from ..transporters.transporter import Command
from .general_data_structure_transporter import GeneralDataStructureTransporter


class PreprocessingTransporter():
    """Preprocessing object dedicated Transporter."""

    def is_preprocessing_module(self, pre_module):
        """
        Check whether the given module is a sklearn Preprocessing module or not.

        :param pre_module: given object
        :type pre_module: any
        :return: bool
        """
        if isinstance(pre_module, dict):
            return check_str_in_iterable("pymilo-preprocessing-type", pre_module) and pre_module["pymilo-preprocessing-type"] in SKLEARN_PREPROCESSING_TABLE.keys()
        return get_sklearn_type(pre_module) in SKLEARN_PREPROCESSING_TABLE.keys()


    def serialize(self, pre_module):
        """
        Serialize Preprocessing object.

        :param pre_module: given sklearn preprocessing module
        :type pre_module: sklearn.preprocessing
        :return: pymilo serialized pre_module
        """
        if self.is_preprocessing_module(pre_module):
            gdst = GeneralDataStructureTransporter()
            gdst.transport(pre_module, Command.SERIALIZE, False)
            return {
                "pymilo-bypass": True, 
                "pymilo-preprocessing-type": get_sklearn_type(pre_module),
                "pymilo-preprocessing-data": pre_module.__dict__
                }
        else:
            raise("This Preprocessing module either doesn't exist in sklearn.preprocessing or is not supported yet.")


    def deserialize(self, serialized_pre_module):
        """
        Deserialize Preprocessing object.

        :param serialized_pre_module: serializezd preprocessing module(by pymilo)
        :type serialized_pre_module: dict
        :return: retrieved associated sklearn.preprocessing module
        """
        if self.is_preprocessing_module(serialized_pre_module):
            data = serialized_pre_module["pymilo-preprocessing-data"]
            retrieved_pre_module = SKLEARN_PREPROCESSING_TABLE[serialized_pre_module["pymilo-preprocessing-type"]]()
            gdst = GeneralDataStructureTransporter()
            for key in data:
                setattr(retrieved_pre_module, key, gdst.deserialize(data, key, ""))
            return retrieved_pre_module
        else:
            raise("This object isn't a pymilo serialized preprocessing module")
