# -*- coding: utf-8 -*-
"""PyMilo BisectingTree(sklearn.cluster._bisect_k_means) object transporter."""
from sklearn.cluster._bisect_k_means import _BisectingTree

from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter
from ..utils.util import is_iterable


class BisectingTreeTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle BisectingTree object."""

    def serialize(self, data, key, model_type):
        """
        Serialize instances of the BisectingTree class.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], _BisectingTree):
            data[key] = self.serialize_bisecting_tree(data[key], GeneralDataStructureTransporter())
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize _BisectingTree fields of the Decision Trees.

        The associated tree_ field of the pymilo serialized model, is extracted through
        it's previously serialized parameters.
        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated JSON file of the ML model generated by pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if isinstance(content,_BisectingTree):
            return self.deserialize_bisecting_tree(content, GeneralDataStructureTransporter())
        else:
            return content

    def serialize_bisecting_tree(self, bisecting_tree, gdst=None):
        """
        Serialize the bisecting_tree object recursively. 

        :param bisecting_tree: the bisecting_tree object which is going to get serialized.
        :type bisecting_tree: dict
        :param gdst: an instance of GeneralDataStructureTransporter class.
        :type gdst: GeneralDataStructureTransporter
        :return: pymilo-serialized bisecting_tree
        """
        if (gdst is None):
            gdst == GeneralDataStructureTransporter()
        data = bisecting_tree.__dict__
        for key, value in data.items():
            if (isinstance(value, _BisectingTree)):
                data[key] = {
                    'pymiloed_value': self.serialize_bisecting_tree(value, gdst),
                    'pymiloed_model_type': "_BisectingTree"
                }
            else:
                data[key] = gdst.serialize(data, key, str(_BisectingTree))
        return data

    def deserialize_bisecting_tree(self, bisecting_tree_obj, gdst=None):
        print("HERE: ", type(bisecting_tree_obj))
        if (gdst is None):
            gdst == GeneralDataStructureTransporter()
        data = bisecting_tree_obj
        for key, value in data.items():
            if (
                is_iterable(value) and
                "pymiloed_model_type" in value and
                    value["pymiloed_model_type"] == "_BisectingTree"):
                data[key] = self.deserialize_bisecting_tree(value["pymiloed_value"], gdst)
            else:
                data[key] = gdst.deserialize(data, key, str(_BisectingTree))

        center = data["center"]
        indices = data["indices"]
        score = data["score"]

        reconstructed_bisecting_tree = _BisectingTree(center, indices, score)

        for item in data.keys():
            setattr(reconstructed_bisecting_tree, item, data[item])
        return reconstructed_bisecting_tree
