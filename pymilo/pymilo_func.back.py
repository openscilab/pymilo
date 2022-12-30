# -*- coding: utf-8 -*-
"""Functions."""
from .pymilo_param import SKLEARN_MODEL_TABLE, NUMPY_TYPE_DICT, KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn._loss.loss import BaseLoss

def get_preprocessed_before_deserialization(model_type, unserializable_key, unserializable_content):
    if(not unserializable_key in KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION.keys()):
        return None # Todo exception
    match unserializable_key:
        case "estimator_":
            return get_deserialized_linear_model(unserializable_content)
        case "loss_function_":
            return get_deserialized_loss_function(unserializable_content)
        case "_base_loss":
            return get_deserialized_base_loss(model_type, unserializable_content)
        case "_label_binarizer":
            return get_deserialized_label_binarizer(unserializable_content)
        case "active_":
            return get_deserialized_active_(unserializable_content)
        case "n_nonzero_coefs_":
            return get_deserialized_n_nonzero_coefs_(unserializable_content)
        case "scores_":
            return get_deserialized_scores_(unserializable_content)
        case _:
            return None

##########
# Estimator LinearRegression inside RANSAC

def get_deserialized_linear_model(content):
    inner_model_type = content["inner-model-type"]
    inner_model_data = content["inner-model-data"]
    return convert_to_sklearn_model(inner_model_type,inner_model_data)
    
##########            
# LossFunction for SGD-Classifier

from sklearn.linear_model._stochastic_gradient import SGDClassifier

def get_deserialized_loss_function(content):
    if(not "loss" in content):
        # Todo abort, exception handling
        return None
    return SGDClassifier(loss= content["loss"])._get_loss_function(content["loss"])

##########
# BaseLoss function in Tweedie regression
# BaseLoss function in Poisson regression
from sklearn.linear_model._glm import TweedieRegressor
from sklearn.linear_model._glm import PoissonRegressor
from sklearn.linear_model._glm import GammaRegressor

def get_deserialized_base_loss(model_type, content):
    match model_type:
        case "TweedieRegressor":
            power, link = content["power"], content["link"]
            return TweedieRegressor(power= power, link= link)._get_loss()
        case "PoissonRegressor":
            return PoissonRegressor()._get_loss()    
        case "GammaRegressor":
            return GammaRegressor()._get_loss()
        case _:
            print("NOT IMPLEMENTED YET")
            return None
##########
# scores dict in logistic regression cv
def get_deserialized_scores_(content):
    black_list_key_values = []
    if(not isinstance(content,dict)):
        return content
    for key in content:
        if(isinstance(content[key],list)):
            content[key] = np.ndarray(content[key])
        if("np-type" in content[key]):
            new_key = NUMPY_TYPE_DICT[content[key]["np-type"]](key)
            new_value = content[key]["key-value"]
            black_list_key_values.append([key, new_key, new_value])

    for black_key_value in black_list_key_values:
        prev_key, new_key, new_value = black_key_value
        del content[prev_key]
        content[new_key] = new_value
    return content 
     
##########   
# weird n_nonzero_coefs_ field in OMP-CV
def get_deserialized_n_nonzero_coefs_(content):
    if(isinstance(content,dict)):
        return NUMPY_TYPE_DICT[content["np-type"]](content['value'])
    else:
        # has be successfully deserialized before
        return content

##########
# active_ array in Lasso Lars
def get_deserialized_active_(content):
    new_list = []
    for item in content:
        if("np-type" in item.keys()):
            new_list.append(NUMPY_TYPE_DICT[item["np-type"]](item['value']))
        else:
            new_list.append(item)
    return new_list

##########
# Label Binarizer for Ridge Classifier(+[CV])
def get_deserialized_label_binarizer(content):
    raw_lb = KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION["_label_binarizer"]()
    for key in content.keys():
        if isinstance(content[key], list):
            content[key] = np.array(content[key])
    for item in content.keys():
        setattr(raw_lb, item, content[item])
    return raw_lb

def get_serialized_label_binarizer(label_binarizer):
    data = label_binarizer.__dict__
    for key in data.keys():
        if isinstance(data[key], np.ndarray):
            data[key] = data[key].tolist()
    return data

##########
# dict serializer for Logistic regression CV
def get_dict_serialized(dictionary):
    black_list_key_values = []
    for key in dictionary.keys():
        if(isinstance(dictionary[key],np.ndarray)):
            dictionary[key] = dictionary[key].tolist()
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

from sklearn.linear_model import LinearRegression

def get_sklearn_data(model):
    data = model.__dict__
    for key in data.keys():
        # Handling numpy infinity, ransac
        if(isinstance(data[key],type(np.inf))):
            if(np.inf == data[key]):
                data[key] = {
                    "np-type": "numpy.infinity",
                    "value": "infinite" # added for compatibility
                }
        # Handling inner LinearModels
        elif(isinstance(data[key],LinearRegression)):
            inner_model = data[key]
            data[key] = {
                "inner-model-data": get_sklearn_data(inner_model),
                "inner-model-type": get_sklearn_type(inner_model)
            }
        # Handling special Loss function of TweedieRegression
        elif(isinstance(data[key], BaseLoss)):
            match get_sklearn_type(model):
                case "TweedieRegressor": 
                    data[key] = {
                        "power": data["power"], 
                        "link": data["link"]
                        }
                case "PoissonRegressor":
                    data[key] = {
                        # nothing for now.
                    }
                case "GammaRegressor":
                    data[key] = {
                        # nothing for now
                    }
                case _:
                    print("ERROR: NOT IMPLEMENTED YET")
        # Handling loss functions, no abstract generic type for these losses(SGD based models)
        elif(
            (get_sklearn_type(model) == "SGDClassifier" and key == "loss_function_") or
            (get_sklearn_type(model) == "SGDOneClassSVM" and key == "loss_function_") or
            (get_sklearn_type(model) == "Perceptron" and key == "loss_function_") or
            (get_sklearn_type(model) == "PassiveAggressiveClassifier" and key == "loss_function_")
            ):
            data[key] = {
                "loss": data["loss"]
            }
        elif isinstance(data[key], dict):
            data[key] = get_dict_serialized(data[key])

        elif isinstance(data[key], np.int32): # unserializable type numpy.int32
            data[key] = { "value": int(data[key]), "np-type": "numpy.int32" }
        elif isinstance(data[key], np.int64):
            data[key] = { "value": int(data[key]), "np-type": "numpy.int64" }

        elif isinstance(data[key],list): # list type which containts unserializable type numpy.int32
            new_list = []
            for item in data[key]:
                if(isinstance(item,np.int32)):
                    new_list.append({"value": int(item), "np-type": "numpy.int32"})
                elif(isinstance(item,np.int64)):
                    new_list.append({"value": int(item), "np-type": "numpy.int64"})
                else:
                    new_list.append(item)
            data[key] = new_list

        elif isinstance(data[key], preprocessing.LabelBinarizer): # object of  unserializable LabelBinarizer class
            data[key] = get_serialized_label_binarizer(data[key])

        elif isinstance(data[key], np.ndarray): # object of  unserializable numpy.ndarray class
            data[key] = data[key].tolist()

    return data

def get_sklearn_version():
    return sklearn.__version__

def get_sklearn_type(model):
    raw_type = type(model)
    return str(raw_type).split(".")[-1][:-2]

def get_concrete_class_type(object):
    raw_type = type(object)
    return str(raw_type).split(".")[-1][:-2]

def convert_from_import_object_to_sklearn_model(import_obj):
    raw_model = SKLEARN_MODEL_TABLE[import_obj.type]()
    data = import_obj.data
    for key in data.keys():
        if(isinstance(data[key], dict)):
            if("np-type" in data[key].keys()):
                data[key] = NUMPY_TYPE_DICT[data[key]["np-type"]](data[key]["value"])

        if(key in KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION.keys()):
            data[key] = get_preprocessed_before_deserialization(import_obj.type, key, data[key])

        elif isinstance(data[key], list):
            data[key] = np.array(data[key])
    for item in data.keys():
        setattr(raw_model, item, data[item])
    return raw_model

## TODO merge with above?!
def convert_to_sklearn_model(model_type, model_content):
    raw_model = SKLEARN_MODEL_TABLE[model_type]()
    data = model_content
    for key in data.keys():
        if(key in KEYS_NEED_PREPROCESSING_BEFORE_DESERIALIZATION.keys()):
            data[key] = get_preprocessed_before_deserialization(model_type, key, data[key])
        elif isinstance(data[key], list):
            data[key] = np.array(data[key])
    for item in data.keys():
        setattr(raw_model, item, data[item])
    return raw_model    

def compare_model_outputs(exported_model_output, imported_model_output, epsilon_error = 10**(-8)):
    if(len(exported_model_output.keys()) != len(imported_model_output.keys())):
        return False # Todo throw exception
    totalError = 0
    for key in exported_model_output.keys():
        if(not(key in imported_model_output.keys())):
            return False # Todo throw exception
        totalError += np.abs(imported_model_output[key]) - np.abs(exported_model_output[key])
    # print(f'totalError: {totalError}')
    return np.abs(totalError) < epsilon_error