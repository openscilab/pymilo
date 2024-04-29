# -*- coding: utf-8 -*-
"""PyMilo Loss function transporter."""
from .transporter import AbstractTransporter
from ..utils.util import is_primitive, check_str_in_iterable
from sklearn.linear_model._stochastic_gradient import SGDClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import GradientBoostingClassifier

from ..pymilo_param import SKLEARN_ENSEMBLE_TABLE, NOT_SUPPORTED
if SKLEARN_ENSEMBLE_TABLE["HistGradientBoostingRegressor"] != NOT_SUPPORTED:
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier

loss_function_dict = {}
sklearn_baseloss_support = False
try:
    from sklearn._loss.loss import BaseLoss
    sklearn_baseloss_support = True
    loss_function_dict["sklearn._loss.loss.BaseLoss"] = BaseLoss
except BaseException:
    pass

hist_baseloss_support = False
try:
    from sklearn.ensemble._hist_gradient_boosting.loss import BaseLoss
    hist_baseloss_support = True
    loss_function_dict["sklearn.ensemble._hist_gradient_boosting.loss.BaseLoss"] = BaseLoss
except BaseException:
    pass

gb_losses_support = False
try:
    from sklearn.ensemble._gb_losses import LossFunction
    gb_losses_support = True
    loss_function_dict["sklearn.ensemble._gb_losses.LossFunction"] = LossFunction
except BaseException:
    pass


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

        if sklearn_baseloss_support:
            if isinstance(data[key], BaseLoss):
                if (
                        model_type == "GradientBoostingRegressor" or
                        model_type == "GradientBoostingClassifier"):
                    data[key] = {
                        "pymilo-ensemble-loss": {
                            "loss-library": "sklearn._loss.loss.BaseLoss",
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
                            "loss-library": "sklearn._loss.loss.BaseLoss",
                            "loss": data["loss"],
                            "constant_hessian": data[key].__dict__["constant_hessian"],
                            "n_trees_per_iteration_": data["n_trees_per_iteration_"],
                            "model_type": model_type,
                        }
                    }

        if gb_losses_support:
            if isinstance(data[key], LossFunction):
                if model_type == "GradientBoostingRegressor":
                    data[key] = {
                        "pymilo-ensemble-loss": {
                            "loss-library": "sklearn.ensemble._gb_losses.LossFunction",
                            "loss": data["loss"],
                            "alpha": data["alpha"],
                            "model_type": model_type,
                        }
                    }
                elif model_type == "GradientBoostingClassifier":
                    data[key] = {
                        "pymilo-ensemble-loss": {
                            "loss-library": "sklearn.ensemble._gb_losses.LossFunction",
                            "len(classes_)": len(data["classes_"]),
                            "loss": data["loss"],
                            "n_classes_": data["n_classes_"],
                            "model_type": model_type,
                        }
                    }

        if hist_baseloss_support:
            if isinstance(data[key], BaseLoss):
                if (
                        model_type == "HistGradientBoostingRegressor" or
                        model_type == "HistGradientBoostingClassifier"):
                    data[key] = {
                        "pymilo-ensemble-loss": {
                            "loss-library": "sklearn.ensemble._hist_gradient_boosting.loss.BaseLoss",
                            "loss": data["loss"],
                            "hessians_are_constant": data[key].__dict__["hessians_are_constant"],
                            "n_threads": data[key].__dict__["n_threads"],
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
            loss_type = ensemble_loss["loss-library"]
            if loss_type == "sklearn._loss.loss.BaseLoss":
                sample_weight = None if ensemble_loss["constant_hessian"] else True
                if model_type == "GradientBoostingRegressor":
                    return GradientBoostingRegressor(
                        loss=ensemble_loss["loss"],
                        alpha=ensemble_loss["alpha"])._get_loss(sample_weight)

                elif model_type == "GradientBoostingClassifier":
                    gbs = GradientBoostingClassifier(loss=ensemble_loss["loss"])
                    gbs.__dict__["n_classes_"] = ensemble_loss["n_classes"]
                    return gbs._get_loss(sample_weight)

                elif model_type == "HistGradientBoostingRegressor":
                    return HistGradientBoostingRegressor(
                        loss=ensemble_loss["loss"])._get_loss(sample_weight)

                elif model_type == "HistGradientBoostingClassifier":
                    n_trees_per_iteration_ = ensemble_loss["n_trees_per_iteration_"]
                    hgbc = HistGradientBoostingClassifier()
                    hgbc.__dict__["n_trees_per_iteration_"] = n_trees_per_iteration_
                    return hgbc._get_loss(sample_weight)

            elif loss_type == "sklearn.ensemble._hist_gradient_boosting.loss.BaseLoss" and model_type in ["HistGradientBoostingRegressor", "HistGradientBoostingClassifier"]:
                sample_weight = None if ensemble_loss["hessians_are_constant"] else True
                if model_type == "HistGradientBoostingRegressor":
                    return HistGradientBoostingRegressor(
                        loss=ensemble_loss["loss"])._get_loss(sample_weight, ensemble_loss["n_threads"])
                elif model_type == "HistGradientBoostingClassifier":
                    n_trees_per_iteration_ = ensemble_loss["n_trees_per_iteration_"]
                    hgbc = HistGradientBoostingClassifier(loss=ensemble_loss["loss"])
                    hgbc.__dict__["n_trees_per_iteration_"] = n_trees_per_iteration_
                    return hgbc._get_loss(sample_weight, ensemble_loss["n_threads"])

            elif loss_type == "sklearn.ensemble._gb_losses.LossFunction" and model_type in ["GradientBoostingRegressor", "GradientBoostingClassifier"]:
                from sklearn.ensemble._gb_losses import MultinomialDeviance, BinomialDeviance, LOSS_FUNCTIONS
                if ensemble_loss["loss"] in ["deviance", "log_loss"]:
                    loss_class = (
                        MultinomialDeviance
                        if ensemble_loss["len(classes_)"] > 2
                        else BinomialDeviance
                    )
                else:
                    loss_class = LOSS_FUNCTIONS[ensemble_loss["loss"]]
                if model_type == "GradientBoostingRegressor":
                    if ensemble_loss["loss"] in ("huber", "quantile"):
                        return loss_class(ensemble_loss["alpha"])
                    else:
                        return loss_class()
                elif model_type == "GradientBoostingClassifier":
                    return loss_class(ensemble_loss["n_classes_"])

        return content
