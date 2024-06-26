# -*- coding: utf-8 -*-
"""PyMilo TreePredictor transporter."""
from sklearn.ensemble._hist_gradient_boosting.predictor import TreePredictor
from ..utils.util import check_str_in_iterable
from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter


class TreePredictorTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle TreePredictor objects."""

    def serialize(self, data, key, model_type):
        """
        Serialize TreePredictor object[useful in HistGradientBoosting(Regressor,Classifier)].

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
        if isinstance(data[key], TreePredictor):
            return self.serialize_tree_predictor(data[key])
        elif isinstance(data[key], list):
            return self.serialize_possible_inner_tree_predictor(data[key])
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize previously pymilo serialized TreePredictor object[useful in HistGradientBoosting(Regressor,Classifier)].

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
        if self.is_serialized_treepredictor(content):
            return self.deserialize_tree_predictor(content)
        if isinstance(content, list):
            return self.deserialize_possible_inner_tree_predictor(content)
        return content

    def is_treepredictor(self, treepredictor):
        """
        Check if the given object is an instance of TreePredictor class.

        :param treepredictor: given object to check
        :type treepredictor: any

        :return: bool
        """
        return isinstance(treepredictor, TreePredictor)

    def is_serialized_treepredictor(self, serialized_treepredictor):
        """
        Check if the given object is a previously pymilo-serialized TreePredictor.

        :param serialized_treepredictor: given object to check
        :type serialized_treepredictor: any

        :return: bool
        """
        return check_str_in_iterable(
            "pymiloed-data-structure",
            serialized_treepredictor) and serialized_treepredictor["pymiloed-data-structure"] == "TreePredictor"

    def serialize_tree_predictor(self, treepredictor):
        """
        Serialize given Treepredictor instance.

        :param treepredictor: given treepredictor to get serialized
        :type treepredictor: Treepredictor

        :return: dict
        """
        gdst = GeneralDataStructureTransporter()
        return {
            "pymilo-bypass": True,
            "pymiloed-data-structure": 'TreePredictor',
            "pymiloed-data": {
                "nodes": gdst.deep_serialize_ndarray(treepredictor.nodes),
                "binned_left_cat_bitsets": gdst.deep_serialize_ndarray(treepredictor.binned_left_cat_bitsets),
                "raw_left_cat_bitsets": gdst.deep_serialize_ndarray(treepredictor.raw_left_cat_bitsets),
            },
        }

    def deserialize_tree_predictor(self, serialized_tree_predictor):
        """
        Deserialize to pure Treepredictor object.

        :param serialized_tree_predictor: pymilo-serialized treepredictor
        :type serialized_tree_predictor: dict

        :return: Treepredictor
        """
        gdst = GeneralDataStructureTransporter()
        nodes = serialized_tree_predictor["pymiloed-data"]["nodes"]["pymiloed-ndarray-list"]
        for idx, value in enumerate(nodes):
            nodes[idx] = tuple(value)

        binned_left_cat_bitsets = serialized_tree_predictor["pymiloed-data"]["binned_left_cat_bitsets"]
        raw_left_cat_bitsets = serialized_tree_predictor["pymiloed-data"]["raw_left_cat_bitsets"]

        return TreePredictor(
            nodes=gdst.deep_deserialize_ndarray(
                serialized_tree_predictor["pymiloed-data"]["nodes"]),
            binned_left_cat_bitsets=gdst.deep_deserialize_ndarray(
                binned_left_cat_bitsets),
            raw_left_cat_bitsets=gdst.deep_deserialize_ndarray(
                raw_left_cat_bitsets)
        )

    def serialize_possible_inner_tree_predictor(self, _list):
        """
        Traverse over list and serialize Treepredictor objects.

        :param _list: given list to serialize inner Treepredictor objects
        :type _list: list

        :return: list
        """
        for idx, value in enumerate(_list):
            if self.is_treepredictor(value):
                _list[idx] = self.serialize_tree_predictor(value)
            if isinstance(value, list):
                _list[idx] = self.serialize_possible_inner_tree_predictor(value)
        return _list

    def deserialize_possible_inner_tree_predictor(self, _list):
        """
        Traverse over list and deserialize previously pymilo-serialized Treepredictor objects.

        :param _list: given list to deserialize inner Treepredictor objects
        :type _list: list

        :return: list
        """
        for idx, value in enumerate(_list):
            if self.is_serialized_treepredictor(value):
                _list[idx] = self.deserialize_tree_predictor(value)
            if isinstance(value, list):
                _list[idx] = self.deserialize_possible_inner_tree_predictor(value)
        return _list
