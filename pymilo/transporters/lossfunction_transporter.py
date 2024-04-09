# -*- coding: utf-8 -*-
"""PyMilo Loss function transporter."""
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn._loss.loss import BaseLoss
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter


class LossFunctionTransporter(AbstractTransporter):
    """Customized PyMilo Transporter developed to handle Loss function field."""

    def serialize(self, data, key, model_type):
        """
        Serialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.

        serialize the data[key] of the given model which type is model_type.
        basically in order to fully serialize a model, we should traverse over all the keys of its data dictionary and
        pass it through the chain of associated transporters to get fully serialized.

        :param data: the internal data dictionary of the given model
        :type data: dict
        :param key: the special key of the data param, which we're going to serialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which data dictionary is given as the data param
        :type model_type: str
        :return: pymilo serialized output of data[key]
        """
        if (
            (model_type == "SGDClassifier" and (key == "loss_function_" or key == "_loss_function_")) or
            (model_type == "SGDOneClassSVM" and (key == "loss_function_" or key == "_loss_function_")) or
            (model_type == "Perceptron" and (key == "loss_function_" or key == "_loss_function_")) or
            (model_type == "PassiveAggressiveClassifier" and (key == "loss_function_" or key == "_loss_function_"))
        ):
            data[key] = {
                "pymilo-sgd-loss": data["loss"]
            }

        if isinstance(data[key], BaseLoss):
            if (
                    model_type == "GradientBoostingRegressor" or
                    model_type == "GradientBoostingClassifier"):
                data[key] = {
                    "pymilo-ensemble-loss": {
                        "loss": data["loss"],
                        "constant_hessian": data[key].__dict__["constant_hessian"],
                        "n_classes": data[key].__dict__["n_classes"],
                        "alpha": data["alpha"],
                        "model_type": model_type,
                    }
                }
            elif (
                    model_type == "HistGradientBoostingRegressor" or
                    model_type == "HistGradientBoostingClassifier"):
                data[key] = {
                    "pymilo-ensemble-loss": {
                        "loss": data["loss"],
                        "constant_hessian": data[key].__dict__["constant_hessian"],
                        "n_trees_per_iteration_": data["n_trees_per_iteration_"],
                        "model_type": model_type,
                    }
                }
        return data[key]

    def deserialize(self, data, key, model_type):
        """
        Deserialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.

        the associated loss_function_ field of the pymilo serialized model, is extracted through
        the SGDClassifier's _get_loss_function function with enough feeding of the needed inputs.

        deserialize the data[key] of the given model which type is model_type.
        basically in order to fully deserialize a model, we should traverse over all the keys of its serialized data dictionary and
        pass it through the chain of associated transporters to get fully deserialized.

        :param data: the internal data dictionary of the associated json file of the ML model which is generated previously by
        pymilo export.
        :type data: dict
        :param key: the special key of the data param, which we're going to deserialize its value(data[key])
        :type key: object
        :param model_type: the model type of the ML model, which internal serialized data dictionary is given as the data param
        :type model_type: str
        :return: pymilo deserialized output of data[key]
        """
        content = data[key]
        if is_primitive(content) or content is None:
            return content

        if check_str_in_iterable("pymilo-sgd-loss", content):
            return SGDClassifier(
                loss=content["pymilo-sgd-loss"])._get_loss_function(
                content["pymilo-sgd-loss"])

        if check_str_in_iterable("pymilo-ensemble-loss", content):
            ensemble_loss = content["pymilo-ensemble-loss"]
            model_type = ensemble_loss["model_type"]

            if model_type == "GradientBoostingRegressor":
                return GradientBoostingRegressor(
                    loss=ensemble_loss["loss"],
                    alpha=ensemble_loss["alpha"])._get_loss(
                    ensemble_loss["constant_hessian"])

            elif model_type == "GradientBoostingClassifier":
                gbs = GradientBoostingClassifier(loss=ensemble_loss["loss"])
                gbs.__dict__["n_classes_"] = ensemble_loss["n_classes"]
                return gbs._get_loss(ensemble_loss["constant_hessian"])

            elif model_type == "HistGradientBoostingRegressor":
                return HistGradientBoostingRegressor(
                    loss=ensemble_loss["loss"])._get_loss(
                    ensemble_loss["constant_hessian"])

            elif model_type == "HistGradientBoostingClassifier":
                n_trees_per_iteration_ = ensemble_loss["n_trees_per_iteration_"]
                hgbc = HistGradientBoostingClassifier()
                hgbc.__dict__["n_trees_per_iteration_"] = n_trees_per_iteration_
                return hgbc._get_loss(ensemble_loss["constant_hessian"])

        return content
