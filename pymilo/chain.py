from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum
from .utils.util import is_primitive, is_iterable, check_string_field_existence_in_an_iterable

class Command(Enum):
    SERIALIZE = 1
    DESERIALZIE = 2

# Transporter Interface
class Transporter(ABC):

    @abstractmethod
    def serialize(self,data,key, model_type):
        pass

    @abstractmethod
    def deserialize(self,data,key, model_type):
        pass

    @abstractmethod
    def transport(self, request, command):
        pass

class AbstractTransporter(Transporter):

    def bypass(self, content):
        if(is_primitive(content)):
            return False

        if(check_string_field_existence_in_an_iterable("by-pass", content)):
            return content["by-pass"]
        else:
            return False 

    def transport(self, request, command, is_inner_model = False):
        if command == Command.SERIALIZE:
            # request is a sklearn model
            data = request.__dict__
            for key in data.keys():
                if(self.bypass(data[key])):
                    continue # by-pass!!
                data[key] = self.serialize(data,key,get_sklearn_type(request))
                
        elif command == Command.DESERIALZIE:
            # request is a pymilo-created import object
            data = None
            model_type = None 
            if(is_inner_model):
                data = request["data"]
                model_type = request["type"]
            else:
                data = request.data
                model_type = request.type
            for key in data.keys():
                data[key] = self.deserialize(data,key,model_type)
            return

        else:
            # TODO error handeling.
            return None
"""
All Concrete Transporters either transport a request or pass it to the next transporter in
the chain.
"""

from .utils.util import get_sklearn_type
from .pymilo_param import NUMPY_TYPE_DICT

import numpy as np
class NumpyOrGeneralDataStructureTransporter(AbstractTransporter):

    # dict serializer for Logistic regression CV
    # change ndarray values to list, save unserializable values of numpy.int32|int64 types in an seriazable custom object form.
    def serialize_dict(self,dictionary):
        black_list_key_values = []
        for key in dictionary.keys():
            # check inner field as a np.ndarray
            if(isinstance(dictionary[key],np.ndarray)):
                dictionary[key] = dictionary[key].tolist()
            # check inner field as np.int32
            if(isinstance(key,np.int32)):
                new_value = {
                    "np-type": "numpy.int32",
                    "key-value": dictionary[key]
                    }
                black_list_key_values.append([key,new_value])
        for black_key_value in black_list_key_values:
            prev_key = black_key_value[0]
            new_value = black_key_value[1]
            del dictionary[prev_key]
            dictionary[int(prev_key)] = new_value
        return dictionary

    def serialize(self, data, key, model_type):
        # 1. Handling numpy infinity, ransac
        if(isinstance(data[key],type(np.inf))):
            if(np.inf == data[key]):
                data[key] = {
                    "np-type": "numpy.infinity",
                    "value": "infinite" # added for compatibility
                }

        # 2. unserializable type numpy.int32
        elif isinstance(data[key], np.int32): 
            data[key] = { "value": int(data[key]), "np-type": "numpy.int32" }

        # 3. unserializable type numpy.int64
        elif isinstance(data[key], np.int64):
            data[key] = { "value": int(data[key]), "np-type": "numpy.int64" }

        # 4. list type which may containts unserializable type numpy.int32 | numpy.int64
        elif isinstance(data[key],list): 
            new_list = []
            for item in data[key]:
                if(isinstance(item,np.int32)):
                    new_list.append({"value": int(item), "np-type": "numpy.int32"})
                elif(isinstance(item,np.int64)):
                    new_list.append({"value": int(item), "np-type": "numpy.int64"})
                else:
                    new_list.append(item)
            data[key] = new_list
        
        # 5. object of  unserializable numpy.ndarray class
        # TODO integrate with above list serialization, may containt np.int32 or np.int64 later
        elif isinstance(data[key], np.ndarray): 
            data[key] = data[key].tolist()

        # 6. dictionary serialization 
        elif isinstance(data[key], dict):
            data[key] = self.serialize_dict(data[key])

        return data[key] 

    def deserialize(self, data, key, model_type):
        if(isinstance(data[key],dict)):
            return self.get_deserialized_dict(data[key])
        elif(isinstance(data[key],list)):
            return self.get_deserialized_list(data[key])
        elif(self.is_numpy_primary_type(data[key])):
            return self.get_deserialized_regular_primary_types(data[key])
        else:
            # TODO
            return data[key]

    # used for scores_ field in Logistic regression([+CV])
    # dict deserializer for Logistic regression CV
    # change list values to ndarray, retrive unserializable values of numpy.int32|int64 types.
    def get_deserialized_dict(self, content):
        black_list_key_values = []
        if(not isinstance(content,dict)):
            return content
        for key in content:
            if(isinstance(content[key],list)):
                content[key] = self.get_deserialized_list(content[key])
            if(check_string_field_existence_in_an_iterable("np-type",content[key])):
                new_key = NUMPY_TYPE_DICT[content[key]["np-type"]](key)
                new_value = content[key]["key-value"]
                black_list_key_values.append([key, new_key, new_value])
        for black_key_value in black_list_key_values:
            prev_key, new_key, new_value = black_key_value
            del content[prev_key]
            content[new_key] = new_value
        return content 

    # active_ array in Lasso Lars
    def get_deserialized_list(self,content):
        if(not(isinstance(content,list))):
            return None 
        new_list = []
        for item in content:
            if(is_primitive(item)):
                new_list.append(item)
            elif("np-type" in item.keys()):
                new_list.append(NUMPY_TYPE_DICT[item["np-type"]](item['value']))
            else:
                new_list.append(item)
        return np.array(new_list)

    def get_deserialized_regular_primary_types(self,content):
        if("np-type" in content):
            return NUMPY_TYPE_DICT[content["np-type"]](content['value'])

    def is_numpy_primary_type(self, content):
        if(is_primitive(content)):
            return False
        current_supported_primary_types = NUMPY_TYPE_DICT.values()
        if(not(is_iterable(content))):
            return False 
        if("np-type" in content and content["np-type"] in current_supported_primary_types):
            return True
        else:
            return False


# Handling BaseLoss function in GLMs.
# BaseLoss function in Tweedie regression
# BaseLoss function in Poisson regression
# BaseLoss function in Gamma regression
from sklearn.linear_model._glm import TweedieRegressor
from sklearn.linear_model._glm import PoissonRegressor
from sklearn.linear_model._glm import GammaRegressor
from sklearn._loss.loss import BaseLoss
class BaseLossObjectTransporter(AbstractTransporter):

    def serialize(self, data, key, model_type):
        # Handling special Loss function of GLMs.
        if(isinstance(data[key], BaseLoss)):
            match model_type:
                case "TweedieRegressor": 
                    data[key] = {
                        "power": data["power"], 
                        "link": data["link"],
                        "pymilo_glm_base_loss": True
                        }
                case "PoissonRegressor":
                    data[key] = {
                        "pymilo_glm_base_loss": True
                        # nothing for now.
                    }
                case "GammaRegressor":
                    data[key] = {
                        "pymilo_glm_base_loss": True
                        # nothing for now
                    }
                case _:
                    # print("ERROR: NOT IMPLEMENTED YET")
                    # TODO
                    return data[key]

        return data[key]

    def get_deserialized_base_loss(self, model_type, content):

        match model_type:
            case "TweedieRegressor":
                if(not("power" in content and "link" in content)):
                    return None # TODO EXCEPTION HANDLING
                power, link = content["power"], content["link"]
                return TweedieRegressor(power= power, link= link)._get_loss()
            case "PoissonRegressor":
                return PoissonRegressor()._get_loss()    
            case "GammaRegressor":
                return GammaRegressor()._get_loss()
            case _:
                # print("NOT IMPLEMENTED YET")
                # TODO
                return content 

    def deserialize(self,data,key,model_type):
        content = data[key]
        if(not(check_string_field_existence_in_an_iterable("pymilo_glm_base_loss", content))):
            return content
        return self.get_deserialized_base_loss(model_type,content)

        
# Handling LossFunction for SGD-Classifier
from sklearn.linear_model._stochastic_gradient import SGDClassifier
class LossFunctionTransporter(AbstractTransporter):

    ## SERIALIZATION
    def serialize(self, data, key, model_type):
        if(
            (model_type == "SGDClassifier" and key == "loss_function_") or
            (model_type == "SGDOneClassSVM" and key == "loss_function_") or
            (model_type == "Perceptron" and key == "loss_function_") or
            (model_type == "PassiveAggressiveClassifier" and key == "loss_function_")
            ):
            data[key] = {
                "loss": data["loss"]
            }
        return data[key]

    ## DESERIALIZATION
    def deserialize(self, data, key, model_type):
        content = data[key]
        if(is_primitive(content) or isinstance(content,type(None))):
            return content 
        if(not(check_string_field_existence_in_an_iterable("loss", content))):
            return content
        return SGDClassifier(loss= content["loss"])._get_loss_function(content["loss"])


# TODO exception handling
# Handling Label Binarizer for Ridge Classifier(+[CV])
from sklearn import preprocessing
from .pymilo_param import KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION
class LabelBinarizerTransporter(AbstractTransporter):

    ## SERIALIZATION
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

    ## DESERIALIZATION 
    def deserialize(self, data, key,model_type):
        content = data[key]
        if(key != "_label_binarizer"):
            return content
        return self.get_deserialized_label_binarizer(content)

    def get_deserialized_label_binarizer(self,content):
        raw_lb = KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION["_label_binarizer"]()
        for key in content.keys():
            if isinstance(content[key], list):
                content[key] = np.array(content[key])
        for item in content.keys():
            setattr(raw_lb, item, content[item])
        return raw_lb









