# -*- coding: utf-8 -*-
"""PyMilo chain for ensemble models."""
from ..transporters.transporter import Command
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.lossfunction_transporter import LossFunctionTransporter
from ..transporters.bunch_transporter import BunchTransporter
from ..transporters.generator_transporter import GeneratorTransporter
from ..transporters.treepredictor_transporter import TreePredictorTransporter
from ..transporters.binmapper_transporter import BinMapperTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter

from ..pymilo_param import SKLEARN_ENSEMBLE_TABLE

from .util import get_concrete_transporter

from ..exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from ..exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes

from ..utils.util import get_sklearn_type, check_str_in_iterable

from numpy import ndarray, asarray
from traceback import format_exc
from ast import literal_eval

import copy

ENSEMBLE_CHAIN = {
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "TreePredictorTransporter": TreePredictorTransporter(),
    "BinMapperTransporter": BinMapperTransporter(),
    "GeneratorTransporter": GeneratorTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
    "LossFunctionTransporter": LossFunctionTransporter(),
    "BunchTransporter": BunchTransporter(),
}


def get_transporter(model):
    """
    Get associated transporter for the given ML model.

    :param model: given model to get it's transporter
    :type model: scikit ML model
    :return: tuple(ML_MODEL_CATEGORY, transporter function)
    """
    if isinstance(model, str):
        if model.upper() == "ENSEMBLE":
            return "ENSEMBLE", transport_ensemble
    if is_ensemble(model):
        return "ENSEMBLE", transport_ensemble
    else:
        return get_concrete_transporter(model)


def is_ensemble(model):
    """
    Check if the input model is a sklearn's ensemble model.

    :param model: is a string name of a ensemble or a sklearn object of it
    :type model: any object
    :return: check result as bool
    """
    if isinstance(model, str):
        return model in SKLEARN_ENSEMBLE_TABLE
    else:
        return get_sklearn_type(model) in SKLEARN_ENSEMBLE_TABLE.keys()


def transport_ensemble(request, command, is_inner_model=False):
    """
    Return the transported (Serialized or Deserialized) model.

    :param request: given ensemble to be transported
    :type request: any object
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :param is_inner_model: determines whether it is an inner ensemble model of a super ml model
    :type is_inner_model: boolean
    :return: the transported request as a json string or sklearn ensemble model
    """
    if not is_inner_model:
        _validate_input(request, command)

    if command == Command.SERIALIZE:
        try:
            return serialize_ensemble(request)
        except Exception as e:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc(),
                    },
                    'object': request,
                })

    elif command == Command.DESERIALIZE:
        try:
            return deserialize_ensemble(request, is_inner_model)
        except Exception as e:
            raise PymiloDeserializationException(
                {
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': request})


def serialize_possible_ml_model(possible_ml_model):
    """
    Check whether the given object is a ML model and if it is, serialize it.

    :param possible_ml_model: given obj to check
    :type possible_ml_model: obj
    :return: tuple(bool, whether itself or dict)
    """
    if isinstance(possible_ml_model, str):
        return False, possible_ml_model
    ml_category, transporter = get_transporter(possible_ml_model)
    if transporter is not None:
        return True, {
            "pymilo-bypass": True,
            "pymilo-inner-model-data": transporter(possible_ml_model, Command.SERIALIZE),
            "pymilo-inner-model-type": get_sklearn_type(possible_ml_model),
            "pymilo-ml-category": ml_category
        }
    else:
        return False, possible_ml_model


def deserialize_possible_ml_model(possible_serialized_ml_model):
    """
    Check whether the given object is previously serialized ML model and if it is, deserialize it back to the associated ML model.

    :param possible_serialized_ml_model: given obj to check
    :type possible_serialized_ml_model: obj
    :return: tuple(bool, whether itself or a scikit ML model)
    """
    if check_str_in_iterable("pymilo-inner-model-type", possible_serialized_ml_model):
        _, transporter = get_transporter(possible_serialized_ml_model["pymilo-ml-category"])
        return True, transporter({
            "data": possible_serialized_ml_model["pymilo-inner-model-data"],
            "type": possible_serialized_ml_model["pymilo-inner-model-type"]
        }, Command.DESERIALIZE, is_inner_model=True)
    else:
        return False, possible_serialized_ml_model


def serialize_ensemble(ensemble_object):
    """
    Return the serialized json string of the given ensemble model.

    :param ensemble_object: given model to be get serialized
    :type ensemble_object: any sklearn ensemble model
    :return: the serialized json string of the given ensemble
    """
    for transporter in ENSEMBLE_CHAIN:
        if transporter != "GeneralDataStructureTransporter":
            ENSEMBLE_CHAIN[transporter].transport(
                ensemble_object, Command.SERIALIZE)

    for key, value in ensemble_object.__dict__.items():
        if isinstance(value, list):
            has_inner_tuple_with_ml_model = False
            pt = PreprocessingTransporter()
            for idx, item in enumerate(value):
                if isinstance(item, tuple):
                    listed_tuple = list(item)
                    for inner_idx, inner_item in enumerate(listed_tuple):
                        if pt.is_preprocessing_module(inner_item):
                            listed_tuple[inner_idx] = pt.serialize_pre_module(inner_item)
                        else:
                            has_inner_model, result = serialize_possible_ml_model(inner_item)
                            if has_inner_model:
                                has_inner_tuple_with_ml_model = True
                            listed_tuple[inner_idx] = result
                    value[idx] = listed_tuple
                else:
                    value[idx] = serialize_possible_ml_model(item)[1]
            if has_inner_tuple_with_ml_model:
                ensemble_object.__dict__[key] = {
                    "pymiloed-data-structure": "list of (str, estimator) tuples",
                    "pymiloed-data": value,
                }

        elif isinstance(value, dict):
            if not check_str_in_iterable("pymilo-bypass", value):
                if check_str_in_iterable(
                    "pymiloed-data-structure",
                        value) and value["pymiloed-data-structure"] == "Bunch":
                    new_value = {}
                    for inner_key, inner_value in value["pymiloed-data"].items():
                        new_value[inner_key] = serialize_possible_ml_model(inner_value)[1]
                    value["pymiloed-data"] = new_value
                else:
                    new_value = {}
                    for inner_key, inner_value in value.items():
                        new_value[inner_key] = serialize_possible_ml_model(inner_value)[1]
                    ensemble_object.__dict__[key] = new_value

        elif isinstance(value, ndarray):
            has_inner_model, result = serialize_models_in_ndarray(value)
            if has_inner_model:
                ensemble_object.__dict__[key] = result

        else:
            ensemble_object.__dict__[key] = serialize_possible_ml_model(value)[1]

    ENSEMBLE_CHAIN["GeneralDataStructureTransporter"].transport(ensemble_object, Command.SERIALIZE)

    return ensemble_object.__dict__


def serialize_models_in_ndarray(ndarray_instance):
    """
    Serialize the ml models inside the given ndarray.

    :param ndarray_instance: given ndarray needed to get it's inner ML models serialized
    :type ndarray_instance: numpy.ndarray
    :return: dict
    """
    if not isinstance(ndarray_instance, ndarray):
        return None  # throw error

    ndarray_instance_copy = copy.deepcopy(ndarray_instance)
    has_inner_model = True

    dtype = ndarray_instance.dtype

    new_list = []
    for item in ndarray_instance:
        if isinstance(item, ndarray):
            has_inside_model, result = serialize_models_in_ndarray(item)
            if not has_inside_model:
                has_inner_model = False
                break
            else:
                new_list.append(result)
        else:
            has_ml_model, result = serialize_possible_ml_model(item)
            if has_ml_model:
                new_list.append(result)
            else:
                has_inner_model = False
                break

    if not has_inner_model:
        return False, ndarray_instance_copy
    else:
        return True, {
            'pymiloed-ndarray-list': new_list,
            'pymiloed-ndarray-dtype': str(dtype),
            'pymiloed-data-structure': 'numpy.ndarray'
        }


def deserialize_models_in_ndarray(serialized_ndarray):
    """
    Deserializes possible ML models within the given ndarray instance.

    :param serialized_ndarray: given ndarray to deserialize possible previously serialized inner ML models
    :type serialized_ndarray: obj
    :return: numpy.ndarray
    """
    gdst = GeneralDataStructureTransporter()
    if not gdst.is_deserialized_ndarray(serialized_ndarray):
        return False, None  # throw error

    serialized_ndarray_copy = copy.deepcopy(serialized_ndarray)
    has_inner_model = True

    inner_list = serialized_ndarray['pymiloed-ndarray-list']
    new_list = []
    for _, item in enumerate(inner_list):
        if gdst.is_deserialized_ndarray(item):
            has_inside_model, result = deserialize_models_in_ndarray(item)
            if not has_inside_model:
                has_inside_model = False
                break
            else:
                new_list.append(result)

        else:
            has_ml_model, result = deserialize_possible_ml_model(item)
            if has_ml_model:
                new_list.append(result)
            else:
                has_inner_model = False
                break

    if not has_inner_model:
        return False, serialized_ndarray_copy
    else:
        dtype = serialized_ndarray['pymiloed-ndarray-dtype']
        if dtype.startswith("["):
            dtype = literal_eval(dtype)

        return True, asarray(new_list, dtype=dtype)


def deserialize_ensemble(ensemble, is_inner_model=False):
    """
    Return the associated sklearn ensemble model of the given ensemble.

    :param ensemble: given json string of a ensemble model to get deserialized to associated sklearn ensemble model
    :type ensemble: obj
    :param is_inner_model: determines whether it is an inner ensemble model of a super ml model
    :type is_inner_model: boolean
    :return: associated sklearn ensemble model
    """
    data = None
    if is_inner_model:
        data = ensemble["data"]
    else:
        data = ensemble.data

    for transporter in ENSEMBLE_CHAIN:
        if transporter != "GeneralDataStructureTransporter":
            ENSEMBLE_CHAIN[transporter].transport(
                ensemble, Command.DESERIALIZE, is_inner_model)

    for key, value in data.items():
        if isinstance(value, dict):
            if check_str_in_iterable("pymiloed-data-structure",
                                     value) and value["pymiloed-data-structure"] == "list of (str, estimator) tuples":
                listed_tuples = value["pymiloed-data"]
                list_of_tuples = []
                pt = PreprocessingTransporter()
                for listed_tuple in listed_tuples:
                    name, serialized_model = listed_tuple
                    retrieved_model = pt.deserialize_pre_module(serialized_model) if pt.is_preprocessing_module(
                        serialized_model) else deserialize_possible_ml_model(serialized_model)[1]
                    list_of_tuples.append(
                        (name, retrieved_model)
                    )
                data[key] = list_of_tuples

            elif GeneralDataStructureTransporter().is_deserialized_ndarray(value):
                has_inner_model, result = deserialize_models_in_ndarray(value)
                if has_inner_model:
                    data[key] = result

        if isinstance(value, list):
            for idx, item in enumerate(value):
                has_ml_model, result = deserialize_possible_ml_model(item)
                if has_ml_model:
                    value[idx] = result

        has_ml_model, result = deserialize_possible_ml_model(value)
        if has_ml_model:
            data[key] = result

    ENSEMBLE_CHAIN["GeneralDataStructureTransporter"].transport(ensemble, Command.DESERIALIZE, is_inner_model)

    _type = None
    raw_model = None
    meta_learnings = ["StackingRegressor", "StackingClassifier", "VotingRegressor", "VotingClassifier"]
    pipeline_models = ["Pipeline"]
    if is_inner_model:
        _type = ensemble["type"]
    else:
        _type = ensemble.type

    if _type in meta_learnings:
        raw_model = SKLEARN_ENSEMBLE_TABLE[_type](estimators=data["estimators"])
    elif _type in pipeline_models:
        raw_model = SKLEARN_ENSEMBLE_TABLE[_type](steps=data["steps"])
    else:
        raw_model = SKLEARN_ENSEMBLE_TABLE[_type]()

    for item in data:
        setattr(raw_model, item, data[item])
    return raw_model


def _validate_input(model, command):
    """
    Check if the provided inputs are valid in relation to each other.

    :param model: a sklearn ensemble model or a json string of it, serialized through the pymilo export
    :type model: obj
    :param command: command to specify whether the request should be serialized or deserialized
    :type command: transporter.Command
    :return: None
    """
    if command == Command.SERIALIZE:
        if is_ensemble(model):
            return
        else:
            raise PymiloSerializationException(
                {
                    'error_type': SerializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
    elif command == Command.DESERIALIZE:
        if is_ensemble(model.type):
            return
        else:
            raise PymiloDeserializationException(
                {
                    'error_type': DeserializationErrorTypes.INVALID_MODEL,
                    'object': model
                }
            )
