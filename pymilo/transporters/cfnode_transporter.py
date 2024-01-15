# -*- coding: utf-8 -*-
"""PyMilo CFnode(from sklearn.cluster._birch) object transporter."""
from sklearn.cluster._birch import _CFNode
from sklearn.cluster._birch import _CFSubcluster

from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter


class CFNodeTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle CFnode object."""

    def __init__(self):
        self.all_cfnodes = set()
        self.retrieved_cfnodes = {}

    def serialize(self, data, key, model_type):
        """
        Serialize data[key] if it is an instance of _CFNode.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        self.all_cfnodes = set()
        if isinstance(data[key], _CFNode):
            data[key] = self.serialize_cfnode(data[key], GeneralDataStructureTransporter())
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize data[key] if it is a pymilo serialized _CFNode object.

        :param data: the internal data dictionary of the associated JSON file of the ML model generated by pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        self.retrieved_cfnodes = {}
        content = data[key]
        if isinstance(content, dict) and "pymilo_model_type" in content and content["pymilo_model_type"] == "_CFNode":
            return self.deserialize_cfnode(content, GeneralDataStructureTransporter())
        else:
            return content

    def serialize_cfnode(self, cfnode, gdst):
        """
        Serialize given _CFnode instance recursively.

        :param cfnode: given _CFnode object to get serialized
        :type cfnode: sklearn.cluster._birch._CFNode
        :param gdst: an instance of GeneralDataStructureTransporter class
        :type gdst: pymilo.transporters.general_data_structure_transporter.GeneralDataStructureTransporter
        :return: dict
        """
        data = cfnode.__dict__
        cfnode_id = self.get_cfnode_id(cfnode)
        data["pymilo_cfnode_id"] = cfnode_id
        self.all_cfnodes.add(cfnode_id)
        for key, value in data.items():
            if (isinstance(value, _CFNode)):
                value_id = self.get_cfnode_id(value)
                if (value_id in self.all_cfnodes):
                    data[key] = {
                        "pymilo_model_type": "_CFNode",
                        "pymilo_cfnode_value": "PYMILO_CFNODE_RECURSION",
                        "pymilo_cfnode_id": value_id,
                    }
                else:
                    data[key] = {
                        "pymilo_model_type": "_CFNode",
                        "pymilo_cfnode_value": self.serialize_cfnode(value, gdst),
                        "pymilo_cfnode_id": value_id,
                    }
            elif (isinstance(value, list) and key == "subclusters_"):
                if len(value) > 0:
                    if isinstance(value[0], _CFSubcluster):
                        data[key] = {"pymilo_model_type": "_CFSubcluster", "pymilo_subclusters_value": [
                            self.serialize_cfsubcluster(cf_subcluster, gdst) for cf_subcluster in value], }
                else:
                    data[key] = gdst.serialize(data, key, str(_CFNode))  # TODO model name
            else:
                data[key] = gdst.serialize(data, key, str(_CFNode))
        return data

    def deserialize_cfnode(self, cfnode_pymiloed_obj, gdst):
        """
        Derialize given serialized object of _CFnode class recursively.

        :param cfnode_pymiloed_obj: given serialized _CFnode object to get deserialized
        :type cfnode_pymiloed_obj: obj
        :param gdst: an instance of GeneralDataStructureTransporter class
        :type gdst: pymilo.transporters.general_data_structure_transporter.GeneralDataStructureTransporter
        :return: sklearn.cluster._birch._CFNode
        """
        # this object is a previously pymiloed cfnode object
        if not cfnode_pymiloed_obj["pymilo_cfnode_id"] in self.retrieved_cfnodes.keys():
            self.retrieved_cfnodes[cfnode_pymiloed_obj["pymilo_cfnode_id"]] = self.get_base_cfnode(cfnode_pymiloed_obj)

        current_cfnode = self.retrieved_cfnodes[cfnode_pymiloed_obj["pymilo_cfnode_id"]]
        # init non left, right and subcluster.
        for key, value in cfnode_pymiloed_obj.items():

            if isinstance(value, dict) and "pymilo_model_type" in value and value["pymilo_model_type"] == "_CFNode":
                if value["pymilo_cfnode_id"] in self.retrieved_cfnodes.keys():
                    cfnode_pymiloed_obj[key] = self.retrieved_cfnodes[value["pymilo_cfnode_id"]]
                    # case of recursion.
                else:
                    new_cfnode = self.deserialize_cfnode(value["pymilo_cfnode_value"], gdst)
                    self.retrieved_cfnodes[value["pymilo_cfnode_id"]] = new_cfnode
                    cfnode_pymiloed_obj[key] = new_cfnode

            elif isinstance(value, dict) and "pymilo_model_type" in value and value["pymilo_model_type"] == "_CFSubcluster":
                # has a >0 length subclusters_ fields.
                cfnode_pymiloed_obj[key] = [self.deserialize_cfsubcluster(
                    subcluster, gdst) for subcluster in value["pymilo_subclusters_value"]]
            else:
                cfnode_pymiloed_obj[key] = gdst.deserialize(cfnode_pymiloed_obj, key, str(_CFNode))

        for key, value in cfnode_pymiloed_obj.items():
            setattr(current_cfnode, key, value)
        return current_cfnode

    def serialize_cfsubcluster(self, cfsubcluster, gdst):
        """
        Serialize given _CFSubcluster instance.

        :param cfsubcluster: given _CFSubcluster object to get serialized
        :type cfsubcluster: sklearn.cluster._birch._CFSubcluster
        :param gdst: an instance of GeneralDataStructureTransporter class
        :type gdst: pymilo.transporters.general_data_structure_transporter.GeneralDataStructureTransporter
        :return: dict
        """
        data = cfsubcluster.__dict__
        for key, value in data.items():
            if (isinstance(value, _CFNode)):
                data[key] = self.serialize_cfnode(value)
            else:
                data[key] = gdst.serialize(data, key, str(_CFSubcluster))
        return data

    def deserialize_cfsubcluster(self, cfsubcluster_pymiloed_obj, gdst):
        """
        Deserialize given serialized object of _CFSubcluster class recursively.

        :param cfsubcluster_pymiloed_obj: given serialized _CFSubcluster object to get deserialized
        :type cfsubcluster_pymiloed_obj: obj
        :param gdst: an instance of GeneralDataStructureTransporter class
        :type gdst: pymilo.transporters.general_data_structure_transporter.GeneralDataStructureTransporter
        :return: sklearn.cluster._birch._CFSubcluster
        """
        for key, value in cfsubcluster_pymiloed_obj.items():
            if isinstance(value, dict) and "pymilo_model_type" in value and value["pymilo_model_type"] == "_CFNode":
                cfsubcluster_pymiloed_obj[key] = self.deserialize_cfnode(value["pymilo_cfnode_value"], gdst)
            else:
                cfsubcluster_pymiloed_obj[key] = gdst.deserialize(cfsubcluster_pymiloed_obj, key, str(_CFSubcluster))

        subcluster_instance = _CFSubcluster()
        for key, value in cfsubcluster_pymiloed_obj.items():
            setattr(subcluster_instance, key, value)
        return subcluster_instance

    def get_cfnode_id(self, cfnode):
        """
        Create a unique id for the given cfnode

        :param cfnode: given _CFnode object to generate it's id.
        :type cfnode: sklearn.cluster._birch._CFNode
        :return: str
        """
        if not isinstance(cfnode, _CFNode):
            return "None"
        else:
            return str(cfnode).split(" at ")[1][:-1]

    def get_base_cfnode(self, cfnode_pymiloed_obj):
        """
        Create a basic _CFNode instance from constructor parameters existing in cfnode_pymiloed_obj

        :param cfnode_pymiloed_obj: given serialized _CFnode object to generate it's basic _CFNode instance
        :type cfnode_pymiloed_obj: sklearn.cluster._birch._CFNode
        :return: _CFNode
        """
        threshold = cfnode_pymiloed_obj["threshold"]
        branching_factor = cfnode_pymiloed_obj["branching_factor"]
        is_leaf = cfnode_pymiloed_obj["is_leaf"]
        n_features = cfnode_pymiloed_obj["n_features"]
        dtype = GeneralDataStructureTransporter().list_to_ndarray(cfnode_pymiloed_obj["init_centroids_"]).dtype
        return _CFNode(
            threshold=threshold,
            branching_factor=branching_factor,
            is_leaf=is_leaf,
            n_features=n_features,
            dtype=dtype,
        )