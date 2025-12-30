# -*- coding: utf-8 -*-
"""PyMilo chain for compose models."""

from ..chains.chain import AbstractChain
from ..transporters.compose_transporter import ComposeTransporter
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
from ..transporters.function_transporter import FunctionTransporter
from ..transporters.transporter import Command
from ..pymilo_param import SKLEARN_COMPOSE_TABLE

COMPOSE_CHAIN = {
    "ComposeTransporter": ComposeTransporter(),
    "GeneralDataStructureTransporter": GeneralDataStructureTransporter(),
    "FunctionTransporter": FunctionTransporter(),
}


class ComposeModelChain(AbstractChain):
    """ComposeModelChain developed to handle sklearn Compose ML model transportation."""

    def serialize(self, compose_object):
        """
        Return the serialized json string of the given compose model.

        :param compose_object: given model to be get serialized
        :type compose_object: any sklearn compose model
        :return: the serialized json string of the given compose model
        """
        # Standard chain traversal: ComposeTransporter handles nested estimators,
        # then other transporters normalize general data structures and functions.
        for transporter in self._transporters:
            self._transporters[transporter].transport(compose_object, Command.SERIALIZE)
        return compose_object.__dict__

    def deserialize(self, compose, is_inner_model=False):
        """
        Return the associated sklearn compose model of the given compose.

        :param compose: given json string of a compose model to get deserialized to associated sklearn compose model
        :type compose: obj
        :param is_inner_model: determines whether it is an inner compose model of a super ml model
        :type is_inner_model: boolean
        :return: associated sklearn compose model
        """
        for transporter in self._transporters:
            self._transporters[transporter].transport(
                compose, Command.DESERIALIZE, is_inner_model)

        data = compose["data"] if is_inner_model else compose.data

        _type = None
        raw_model = None
        if is_inner_model:
            _type = compose["type"]
        else:
            _type = compose.type

        # Create the appropriate compose model with required parameters
        if _type == "ColumnTransformer":
            # Extract transformers from the deserialized data
            transformers = data.get("transformers", [])
            remainder = data.get("remainder", "drop")
            sparse_threshold = data.get("sparse_threshold", 0.3)
            n_jobs = data.get("n_jobs", None)
            transformer_weights = data.get("transformer_weights", None)
            verbose = data.get("verbose", False)
            verbose_feature_names_out = data.get("verbose_feature_names_out", True)
            
            raw_model = self._supported_models[_type](
                transformers=transformers,
                remainder=remainder,
                sparse_threshold=sparse_threshold,
                n_jobs=n_jobs,
                transformer_weights=transformer_weights,
                verbose=verbose,
                verbose_feature_names_out=verbose_feature_names_out
            )
        elif _type == "TransformedTargetRegressor":
            # Extract parameters from the deserialized data
            regressor = data.get("regressor", None)
            transformer = data.get("transformer", None)
            func = data.get("func", None)
            inverse_func = data.get("inverse_func", None)
            check_inverse = data.get("check_inverse", True)
            
            raw_model = self._supported_models[_type](
                regressor=regressor,
                transformer=transformer,
                func=func,
                inverse_func=inverse_func,
                check_inverse=check_inverse
            )
        else:
            raw_model = self._supported_models[_type]()

        # Set all other attributes on the raw model
        for item in data:
            if item not in ["transformers", "remainder", "sparse_threshold", "n_jobs", "transformer_weights", "verbose", "verbose_feature_names_out", "regressor", "transformer", "func", "inverse_func", "check_inverse"]:
                setattr(raw_model, item, data[item])
        return raw_model


compose_chain = ComposeModelChain(COMPOSE_CHAIN, SKLEARN_COMPOSE_TABLE)


def get_transporter(model):
    """
    Get associated transporter for the given ML model.

    :param model: given model to get it's transporter
    :type model: scikit ML model
    :return: tuple(ML_MODEL_CATEGORY, transporter function)
    """
    if isinstance(model, str):
        if model.upper() == "COMPOSE":
            return "COMPOSE", compose_chain.transport
    if compose_chain.is_supported(model):
        return "COMPOSE", compose_chain.transport
    else:
        return None, None
