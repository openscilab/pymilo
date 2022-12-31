import numpy as np
from ..pymilo_param import NUMPY_TYPE_DICT
from ..utils.util import is_primitive, is_iterable, check_str_in_iterable
from .transporter import AbstractTransporter


class GeneralDataStructureTransporter(AbstractTransporter):

    # dict serializer for Logistic regression CV
    # change ndarray values to list, save unserializable values of
    # numpy.int32|int64 types in an seriazable custom object form.
    def serialize_dict(self, dictionary):
        black_list_key_values = []
        for key in dictionary.keys():
            # check inner field as a np.ndarray
            if isinstance(dictionary[key], np.ndarray):
                dictionary[key] = dictionary[key].tolist()
            # check inner field as np.int32
            if isinstance(key, np.int32):
                new_value = {
                    "np-type": "numpy.int32",
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
        # 1. Handling numpy infinity, ransac
        if isinstance(data[key], type(np.inf)):
            if (np.inf == data[key]):
                data[key] = {
                    "np-type": "numpy.infinity",
                    "value": "infinite"  # added for compatibility
                }

        # 2. unserializable type numpy.int32
        elif isinstance(data[key], np.int32):
            data[key] = {"value": int(data[key]), "np-type": "numpy.int32"}

        # 3. unserializable type numpy.int64
        elif isinstance(data[key], np.int64):
            data[key] = {"value": int(data[key]), "np-type": "numpy.int64"}

        # 4. list type which may containts unserializable type numpy.int32 |
        # numpy.int64
        elif isinstance(data[key], list):
            new_list = []
            for item in data[key]:
                if (isinstance(item, np.int32)):
                    new_list.append(
                        {"value": int(item), "np-type": "numpy.int32"})
                elif (isinstance(item, np.int64)):
                    new_list.append(
                        {"value": int(item), "np-type": "numpy.int64"})
                else:
                    new_list.append(item)
            data[key] = new_list

        # 5. object of  unserializable numpy.ndarray class
        # TODO integrate with above list serialization, may containt np.int32
        # or np.int64 later
        elif isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()

        # 6. dictionary serialization
        elif isinstance(data[key], dict):
            data[key] = self.serialize_dict(data[key])

        return data[key]

    def deserialize(self, data, key, model_type):
        if isinstance(data[key], dict):
            return self.get_deserialized_dict(data[key])
        elif isinstance(data[key], list):
            return self.get_deserialized_list(data[key])
        elif self.is_numpy_primary_type(data[key]):
            return self.get_deserialized_regular_primary_types(data[key])
        else:
            # TODO
            return data[key]

    # used for scores_ field in Logistic regression([+CV])
    # dict deserializer for Logistic regression CV
    # change list values to ndarray, retrive unserializable values of
    # numpy.int32|int64 types.
    def get_deserialized_dict(self, content):
        black_list_key_values = []
        if not isinstance(content, dict):
            return content
        for key in content:
            if isinstance(content[key], list):
                content[key] = self.get_deserialized_list(content[key])
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

    # active_ array in Lasso Lars
    def get_deserialized_list(self, content):
        if not (isinstance(content, list)):
            return None
        new_list = []
        for item in content:
            if is_primitive(item):
                new_list.append(item)
            elif "np-type" in item.keys():
                new_list.append(
                    NUMPY_TYPE_DICT[item["np-type"]](item['value']))
            else:
                new_list.append(item)
        return np.array(new_list)

    def get_deserialized_regular_primary_types(self, content):
        if "np-type" in content:
            return NUMPY_TYPE_DICT[content["np-type"]](content['value'])

    def is_numpy_primary_type(self, content):
        if is_primitive(content):
            return False
        current_supported_primary_types = NUMPY_TYPE_DICT.values()
        if not (is_iterable(content)):
            return False
        if ("np-type" in content and content["np-type"]
                in current_supported_primary_types):
            return True
        else:
            return False
