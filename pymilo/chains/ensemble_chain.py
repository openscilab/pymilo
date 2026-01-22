# -*- coding: utf-8 -*-
"""PyMilo chain for ensemble models."""

import copy
from ast import literal_eval

from numpy import ndarray, asarray

from ..chains.chain import AbstractChain
from ..transporters.feature_extraction_transporter import FeatureExtractorTransporter
from ..transporters.binmapper_transporter import BinMapperTransporter
from ..transporters.bunch_transporter import BunchTransporter
from ..transporters.transporter import Command
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.generator_transporter import GeneratorTransporter
from ..transporters.lossfunction_transporter import LossFunctionTransporter
from ..transporters.preprocessing_transporter import PreprocessingTransporter
from ..transporters.randomstate_transporter import RandomStateTransporter
from ..transporters.treepredictor_transporter import TreePredictorTransporter
from ..pymilo_param import SKLEARN_ENSEMBLE_TABLE
from ..utils.util import check_str_in_iterable
from .util import serialize_possible_ml_model, deserialize_possible_ml_model

ENSEMBLE_CHAIN = {
    "FeatureExtractorTransporter": FeatureExtractorTransporter(),
    "PreprocessingTransporter": PreprocessingTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "TreePredictorTransporter": TreePredictorTransporter(),
    "BinMapperTransporter": BinMapperTransporter(),
    "GeneratorTransporter": GeneratorTransporter(),
    "RandomStateTransporter": RandomStateTransporter(),
    "LossFunctionTransporter": LossFunctionTransporter(),
    "BunchTransporter": BunchTransporter(),
}


class EnsembleModelChain(AbstractChain):
    """EnsembleModelChain developed to handle sklearn Ensemble ML model transportation."""

    def serialize(self, ensemble_object):
        """
        Return the serialized json string of the given ensemble model.

        :param ensemble_object: given model to be get serialized
        :type ensemble_object: any sklearn ensemble model
        :return: the serialized json string of the given ensemble
        """
        for transporter in self._transporters:
            if transporter != "GeneralDataStructureTransporter":
                self._transporters[transporter].transport(
                    ensemble_object, Command.SERIALIZE)

        pt = ENSEMBLE_CHAIN["PreprocessingTransporter"]
        fe = ENSEMBLE_CHAIN["FeatureExtractorTransporter"]
        for key, value in ensemble_object.__dict__.items():
            if isinstance(value, list):
                has_inner_tuple_with_ml_model = False
                for idx, item in enumerate(value):
                    if isinstance(item, tuple):
                        listed_tuple = list(item)
                        for inner_idx, inner_item in enumerate(listed_tuple):
                            if pt.is_preprocessing_module(inner_item):
                                listed_tuple[inner_idx] = pt.serialize_pre_module(inner_item)
                            elif fe.is_fe_module(inner_item):
                                listed_tuple[inner_idx] = fe.serialize_fe_module(inner_item)
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
                if check_str_in_iterable("pymilo-bunch", value):
                    new_value = {}
                    for inner_key, inner_value in value["pymilo-bunch"].items():
                        new_value[inner_key] = serialize_possible_ml_model(inner_value)[1]
                    value["pymilo-bunch"] = new_value
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

        self._transporters["GeneralDataStructureTransporter"].transport(ensemble_object, Command.SERIALIZE)

        return ensemble_object.__dict__

    def deserialize(self, ensemble, is_inner_model=False):
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

        for transporter in self._transporters:
            if transporter != "GeneralDataStructureTransporter":
                self._transporters[transporter].transport(
                    ensemble, Command.DESERIALIZE, is_inner_model)

        pt = ENSEMBLE_CHAIN["PreprocessingTransporter"]
        fe = ENSEMBLE_CHAIN["FeatureExtractorTransporter"]
        for key, value in data.items():
            if isinstance(value, dict):
                if check_str_in_iterable("pymiloed-data-structure",
                                         value) and value["pymiloed-data-structure"] == "list of (str, estimator) tuples":
                    listed_tuples = value["pymiloed-data"]
                    list_of_tuples = []
                    for listed_tuple in listed_tuples:
                        name, serialized_model = listed_tuple
                        retrieved_model = None
                        if pt.is_preprocessing_module(serialized_model):
                            retrieved_model = pt.deserialize_pre_module(serialized_model)
                        elif fe.is_fe_module(serialized_model):
                            retrieved_model = fe.deserialize_fe_module(serialized_model)
                        else:
                            retrieved_model = deserialize_possible_ml_model(serialized_model)[1]
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

        self._transporters["GeneralDataStructureTransporter"].transport(ensemble, Command.DESERIALIZE, is_inner_model)

        _type = None
        raw_model = None
        meta_learnings = ["StackingRegressor", "StackingClassifier", "VotingRegressor", "VotingClassifier"]
        pipeline_models = ["Pipeline"]
        if is_inner_model:
            _type = ensemble["type"]
        else:
            _type = ensemble.type

        if _type in meta_learnings:
            raw_model = self._supported_models[_type](estimators=data["estimators"])
        elif _type in pipeline_models:
            raw_model = self._supported_models[_type](steps=data["steps"])
        else:
            raw_model = self._supported_models[_type]()

        for item in data:
            setattr(raw_model, item, data[item])
        return raw_model


ensemble_chain = EnsembleModelChain(ENSEMBLE_CHAIN, SKLEARN_ENSEMBLE_TABLE)


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
