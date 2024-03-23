# -*- coding: utf-8 -*-
"""PyMilo GeneralDataStructure transporter."""
import numpy as np
from ast import literal_eval

from ..pymilo_param import NUMPY_TYPE_DICT

from ..utils.util import get_homogeneous_type, all_same
from ..utils.util import is_primitive, is_iterable, check_str_in_iterable

from .transporter import AbstractTransporter


class GeneralDataStructureTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle fields with general datastructures."""

    def serialize_tuple(self, tuple_field):
        """
        Check for non-serializable fields in tuple and serialize them.

            1. Serialize inner np.ndarray fields in tuple

        :param tuple_field: given tuple
        :type tuple_field: tuple
        :return: serializable tuple
        """
        new_tuple = tuple()
        for item in tuple_field:
            if (isinstance(item, np.ndarray)):
                new_tuple += (self.deep_serialize_ndarray(item),)
            else:
                new_tuple += (item,)
        return new_tuple

    # dict serializer for Logistic regression CV
    def serialize_dict(self, dictionary):
        """
        Make all the fields of the given dictionary serializable.

            1. Changing ndarray values to list,
            2. save unserializable values of numpy.int32|int64 types in an serializable custom object form.

        :param dictionary: given dictionary
        :type dictionary: dict
        :return: fully serializable dictionary
        """
        black_list_key_values = []
        for key in dictionary:
            # check inner field as a np.ndarray
            if isinstance(dictionary[key], np.ndarray):
                dictionary[key] = self.deep_serialize_ndarray(dictionary[key])
            # check inner field as np.int32
            if isinstance(key, np.int32):
                new_value = {
                    "np-type": "numpy.int32",
                    "key-value": dictionary[key]
                }
                black_list_key_values.append([key, new_value])
            if isinstance(key, np.int64):
                new_value = {
                    "np-type": "numpy.int64",
                    "key-value": dictionary[key]
                }
                black_list_key_values.append([key, new_value])
        for black_key_value in black_list_key_values:
            prev_key = black_key_value[0]
            new_value = black_key_value[1]
            del dictionary[prev_key]
            dictionary[int(prev_key)] = new_value
        return dictionary

    def serialize(self, data, key, model_type):
        """
        Serialize the general datastructures.

            1. handling numpy infinity(which is an issue in ransac model)
            2. unserializable type numpy.int32
            3. unserializable type numpy.int64
            4. list type which may contain unserializable type numpy.int32|int64
            5. object of  unserializable numpy.ndarray class
            6. dictionary serialization
            7. tuple serialization

        Serialize the data[key] of the given model which its type is model_type.
        basically in order to fully serialize a model, we should traverse over all the keys of its data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which its data dictionary is given as the data param.
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        # 1. Handling numpy infinity, ransac
        if isinstance(data[key], type(np.inf)):
            if np.inf == data[key]:
                data[key] = {
                    "np-type": "numpy.infinity",
                    "value": "infinite"  # added for compatibility
                }

        elif isinstance(data[key], np.intc):
            data[key] = {"value": int(data[key]), "np-type": "numpy.intc"}

        elif isinstance(data[key], np.int32):
            data[key] = {"value": int(data[key]), "np-type": "numpy.int32"}

        elif isinstance(data[key], np.int64):
            data[key] = {"value": int(data[key]), "np-type": "numpy.int64"}

        elif isinstance(data[key], list):
            new_list = []
            for item in data[key]:
                if isinstance(item, np.int32):
                    new_list.append(
                        {"value": int(item), "np-type": "numpy.int32"})
                elif isinstance(item, np.int64):
                    new_list.append(
                        {"value": int(item), "np-type": "numpy.int64"})
                elif isinstance(item, np.ndarray):
                    new_list.append(self.deep_serialize_ndarray(item))
                else:
                    new_list.append(item)
            data[key] = new_list

        elif isinstance(data[key], np.ndarray):
            data[key] = self.deep_serialize_ndarray(data[key])

        elif isinstance(data[key], dict):
            data[key] = self.serialize_dict(data[key])

        elif isinstance(data[key], tuple):
            data[key] = self.serialize_tuple(data[key])

        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize the general datastructures.

            1. Dictionary deserialization
            2. Deep Convertion of lists to numpy.ndarray class
            3. Convert custom serializable object of np.int32|int64 to the main np.int32|int64 type

        deserialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.
        the associated loss_function_ field of the pymilo serialized model, is extracted through the SGDClassifier's _get_loss_function function
        with enough feeding of the needed inputs.

        deserialize the data[key] of the given model which its type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by
        pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which its internal serialized data dictionary is given as the data param.
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        if isinstance(data[key], dict):
            if 'pymilo-bypass' in data[key]:
                return data[key]
            else:
                return self.get_deserialized_dict(data[key])

        elif isinstance(data[key], list):
            new_list = []
            for item in data[key]:
                if self.is_deserialized_ndarray(item):
                    new_list.append(self.deep_deserialize_ndarray(item))
                else:
                    new_list.append(self.deserialize_primitive_type(item))
            return new_list

        elif self.is_numpy_primary_type(data[key]):
            return self.get_deserialized_regular_primary_types(data[key])
        else:
            # TODO
            return data[key]

    def get_deserialized_dict(self, content):
        """
        Deserialize the given previously made serializable dictionary.

            1. convert numpy types values which previously made serializable to its origianl form
            2. deep convertion of list values to nd arrays

        It is mainly used in serializing/deserialzing the "scores_" field in Logistic regression([+CV]).

        :param content: given dictionary
        :type content: dict
        :return: the original dictionary
        """
        black_list_key_values = []

        if not isinstance(content, dict):
            return content

        if self.is_deserialized_ndarray(content):
            return self.deep_deserialize_ndarray(content)

        for key in content:

            if isinstance(content[key], dict):
                content[key] = self.get_deserialized_dict(content[key])

            elif isinstance(content[key], list):
                new_list = []
                for item in content[key]:
                    if self.is_deserialized_ndarray(item):
                        new_list.append(self.deep_deserialize_ndarray(item))
                    else:
                        new_list.append(self.deserialize_primitive_type(item))
                content[key] = new_list

            if check_str_in_iterable(
                    "np-type", content[key]):
                new_key = NUMPY_TYPE_DICT[content[key]["np-type"]](key)
                new_value = content[key]["key-value"]
                black_list_key_values.append([key, new_key, new_value])
        for black_key_value in black_list_key_values:
            prev_key, new_key, new_value = black_key_value
            del content[prev_key]
            content[new_key] = new_value

        return content

    def get_deserialized_list(self, content):
        """
        Deserialize the given list to its original form.

            1. convert previously made serializable numpy types to its original form
            2. convert list to nd array

        It is mainly used in serializing/deserialzing the "active_" array field in Lasso Lars.

        :param content: given list to get
        :type content: list
        :return: the original list
        """
        if not isinstance(content, list):
            return None
        new_list = []
        for item in content:
            new_list.append(self.deserialize_primitive_type(item))
        return np.array(new_list)

    def get_deserialized_regular_primary_types(self, content):
        """
        Deserialize the given item to its original form.

            1. handling np.int32 type
            2. handling np.int64 type
            3. handling np.infinity type

        :param content: given item needed to get back to its original form
        :type content: object
        :return: the associated np.int32|np.int64|np.inf
        """
        if "np-type" in content:
            return NUMPY_TYPE_DICT[content["np-type"]](content['value'])

    def is_numpy_primary_type(self, content):
        """
        Check whether the given object is a numpy primary type.

        :type content: given object to get checked whether it is a numpy primary type or not
        :return: boolean representing whether the associated content is a numpy primary type or not
        """
        if is_primitive(content):
            return False
        current_supported_primary_types = NUMPY_TYPE_DICT.values()
        if not is_iterable(content):
            return False
        if "np-type" in content and content["np-type"] in current_supported_primary_types:
            return True
        else:
            return False

    def ndarray_to_list(self, ndarray_item):
        """
        Convert the given ndarray to its fully listed format.

            1. convert itself to a list
            2. iterate over it's elements and apply ndarray to list conversion if it's eligible

        :param ndarray_item: given ndarray needed to get converted to it's fully listed form
        :type ndarray_item: numpy.ndarray
        :return: list
        """
        if isinstance(ndarray_item, np.ndarray):
            listed_ndarray = ndarray_item.tolist()
            new_list = []
            for item in listed_ndarray:
                new_list.append(self.ndarray_to_list(item))
            return new_list
        else:
            return ndarray_item

    def list_to_ndarray(self, list_item):
        """
        Convert the given list to its fully ndarray format.

            1. iterate over it's elements and apply list to ndarray conversion if it's eligible
            2. convert the coarse-grained list to ndarray

        :param list_item: given list needed to get converted to it's np.ndarray form
        :type list_item: list
        :return: numpy.ndarray
        """
        if isinstance(list_item, list):

            if len(list_item) == 0:
                return np.asarray(list_item)

            new_list = []
            for item in list_item:
                new_list.append(self.list_to_ndarray(item))

            is_homogeneous_type, the_homogeneous_type = get_homogeneous_type(
                new_list)
            if is_homogeneous_type:
                if the_homogeneous_type in [int, float, str, bool]:
                    return np.asarray(new_list)
                elif the_homogeneous_type == np.ndarray:
                    is_homogeneous_type, _ = get_homogeneous_type(
                        [x.dtype for x in new_list])
                    if (is_homogeneous_type):
                        if all_same([len(x) for x in new_list]):
                            try:
                                return np.asarray(new_list)
                            except Exception as _:
                                # when we have a list of ndarrays with different shapes.
                                return new_list

            return np.asarray(new_list, dtype=object)
        else:
            return self.deserialize_primitive_type(list_item)

    def deserialize_primitive_type(self, primitive):
        """
        Deserialize the given primitive data type.

        :param primitive: given primitive needed to get deserialized to it's pure primitive form
        :type primitive: pure python primitive or dict
        :return: pure python primitive or numpy primitive data type
        """
        if is_primitive(primitive):
            return primitive
        elif check_str_in_iterable("np-type", primitive):
            return NUMPY_TYPE_DICT[primitive["np-type"]
                                   ](primitive['value'])
        else:
            return primitive

    def deep_serialize_ndarray(self, ndarray):
        """
        Serialize the given ndarray.

        :param ndarray_item: given ndarray needed to get serialized to
        :type ndarray_item: numpy.ndarray
        :return: dict
        """
        if (not (isinstance(ndarray, np.ndarray))):
            return None  # throw error

        listed_ndarray = ndarray.tolist()
        dtype = ndarray.dtype

        new_list = []
        for item in listed_ndarray:
            if isinstance(item, np.ndarray):
                new_list.append(self.deep_serialize_ndarray(item))
            else:
                new_list.append(item)

        return {
            'pymiloed-ndarray-list': new_list,
            'pymiloed-ndarray-dtype': str(dtype),
            'pymiloed-data-structure': 'numpy.ndarray'
        }

    def is_deserialized_ndarray(self, deserialized_ndarray):
        """
        Check whether the given input is a previously pymilo-deserialized ndarray.

        :param deserialized_ndarray: given input to get checked
        :type deserialized_ndarray: obj
        :return: bool
        """
        if not (isinstance(deserialized_ndarray, dict)):
            return False

        if not (
                'pymiloed-data-structure' in deserialized_ndarray and deserialized_ndarray['pymiloed-data-structure'] == 'numpy.ndarray'):
            return False

        return True

    def deep_deserialize_ndarray(self, deserialized_ndarray):
        """
        Deserialize the given deserialized_ndarray to its fully ndarray format.

        :param deserialized_ndarray: given deserialized_ndarray needed to get deserialized to it's np.ndarray form
        :type deserialized_ndarray: dict
        :return: numpy.ndarray
        """
        if not self.is_deserialized_ndarray(deserialized_ndarray):
            return None  # throw error

        inner_list = deserialized_ndarray['pymiloed-ndarray-list']
        dtype = deserialized_ndarray['pymiloed-ndarray-dtype']

        if dtype.startswith("["):
            dtype = literal_eval(dtype)

        new_list = []
        for item in inner_list:
            if self.is_deserialized_ndarray(item):
                new_list.append(self.deep_deserialize_ndarray(item))
            else:
                new_list.append(item)

        return np.asarray(new_list, dtype=dtype)
