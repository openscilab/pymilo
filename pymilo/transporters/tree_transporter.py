# -*- coding: utf-8 -*-
"""PyMilo SGDOptimizer object transporter."""
from sklearn.tree._tree import Tree

from .transporter import AbstractTransporter
from .general_data_structure_transporter import GeneralDataStructureTransporter
from ..pymilo_param import NUMPY_TYPE_DICT

import numpy as np
import ctypes
import platform

class TreeTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle (pyi,pyx) Tree object."""

    def serialize(self, data, key, model_type):
        """
        Serialize instances of the Tree class.

        Record the n_features, n_classes and n_outputs fields of tree object.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if isinstance(data[key], Tree):
            gdst = GeneralDataStructureTransporter()
            tree = data[key]
            tree_inner_state = tree.__getstate__()

            data[key] = {
                'params': {
                    'internal_state': {
                        "max_depth": tree_inner_state["max_depth"],
                        "node_count": tree_inner_state["node_count"],
                        "nodes": {
                            "types": [str(np.dtype(i).name) for i in tree_inner_state["nodes"][0]],
                            "field-names": list(tree_inner_state["nodes"][0].dtype.names),
                            "values": [node.tolist() for node in tree_inner_state["nodes"]],
                            },
                        "values":  gdst.ndarray_to_list(tree_inner_state["values"]),
                    },
                    'n_features': tree.n_features,
                    'n_classes': gdst.ndarray_to_list(tree.n_classes),
                    'n_outputs': tree.n_outputs,
                }
            }

        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize the special tree_ field of the Decision Trees.

        The associated tree_ field of the pymilo serialized model, is extracted through
        it's previously serialized parameters.

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by
        pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]

        if (key == "tree_" and (model_type == "DecisionTreeRegressor")):
            gdst = GeneralDataStructureTransporter()
            tree_params = content['params']

            tree_internal_state = tree_params["internal_state"]
            
            nodes_dtype_spec = []
            for i in range(len(tree_internal_state["nodes"]["types"])):
                nodes_dtype_spec.append((tree_internal_state["nodes"]["field-names"][i], NUMPY_TYPE_DICT["numpy." + tree_internal_state["nodes"]["types"][i]]))
            nodes = [tuple(node) for node in tree_internal_state["nodes"]["values"]]
            nodes = np.array(nodes, dtype=nodes_dtype_spec)
            
            tree_internal_state = {
                "max_depth": tree_internal_state["max_depth"],
                "node_count": tree_internal_state["node_count"],
                "nodes": nodes,
                "values": gdst.list_to_ndarray(tree_internal_state["values"]),
            }
            
            os_name = platform.system(),
            python_version = platform.python_version()
            _tree = None 

            def extract_python_main_version(python_full_version):
                """
                Extracts main python version from it's full version str.

                Assume the full python version is A.B.C then this function will extract the
                A.B part.

                :param python_full_version: the whole python version, returned by platform.python_version()
                :type python_full_version: str
                :return: the main python version
                """
                first_dot_index = python_full_version.index('.')
                second_dot_index = python_full_version.index('.', first_dot_index + 1)
                python_main_version = python_full_version[0: second_dot_index]
                print("main python version: ", python_main_version)
                return python_main_version

            if os_name == "Windows" and extract_python_main_version(python_version) == "3.6":
                _tree = Tree(
                    ctypes.c_size_t(tree_params["n_features"]),
                    GeneralDataStructureTransporter().list_to_ndarray(tree_params["n_classes"]),
                    ctypes.c_size_t(tree_params["n_outputs"])
                )
            else:
                _tree = Tree(
                    tree_params["n_features"],
                    GeneralDataStructureTransporter().list_to_ndarray(tree_params["n_classes"]),
                    tree_params["n_outputs"]
                )

            _tree.__setstate__(tree_internal_state)

            return _tree 
        
        else:
            return content
