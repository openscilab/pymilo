from sklearn.linear_model._stochastic_gradient import SGDClassifier
from ..utils.util import is_primitive, check_str_in_iterable
from .transporter import AbstractTransporter

# Handling LossFunction for SGD-Classifier


class LossFunctionTransporter(AbstractTransporter):

    # SERIALIZATION
    def serialize(self, data, key, model_type):
        """
        serialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.
        """
        if (
            (model_type == "SGDClassifier" and key == "loss_function_") or
            (model_type == "SGDOneClassSVM" and key == "loss_function_") or
            (model_type == "Perceptron" and key == "loss_function_") or
            (model_type == "PassiveAggressiveClassifier" and key == "loss_function_")
        ):
            data[key] = {
                "loss": data["loss"]
            }
        return data[key]

    # DESERIALIZATION
    def deserialize(self, data, key, model_type):
        """
        deserialize the special loss_function_ of the SGDClassifier, SGDOneClassSVM, Perceptron and PassiveAggressiveClassifier.
        the associated loss_function_ field of the pymilo serialized model, is extracted through the SGDClassifier's _get_loss_function function 
        with enough feeding of the needed inputs.
        """
        content = data[key]
        if is_primitive(content) or isinstance(content, type(None)):
            return content
        if not check_str_in_iterable("loss", content):
            return content
        return SGDClassifier(
            loss=content["loss"])._get_loss_function(
            content["loss"])
