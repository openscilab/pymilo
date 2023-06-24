# -*- coding: utf-8 -*-
"""PyMilo modules."""
from .pymilo_func import get_sklearn_data, get_sklearn_version, to_sklearn_model
from .utils.util import get_sklearn_type
from .pymilo_param import PYMILO_VERSION
import json

from pymilo.exceptions.deserialize_exception import PymiloDeserializationException, DeSerilaizatoinErrorTypes
from pymilo.exceptions.serialize_exception import PymiloSerializationException, SerilaizatoinErrorTypes
from traceback import format_exc


class Export:
    """
    The Pymilo Export class facilitates exporting of models to json files.

    >>> exported_model = Export(model) # the model could be any sklearn linear model.
    >>> exported_model_serialized_path = os.path.join(os.getcwd(), "MODEL_NAME.json")
    >>> exported_model.save(exported_model_serialized_path)
    """

    def __init__(self, model):
        """
        Initialize the Pymilo Export instance.

        :param model: given model(any sklearn linear model)
        :type model: any class of the sklearn's linear models
        :return: an instance of the Pymilo Export class
        """
        self.data = get_sklearn_data(model)
        self.version = get_sklearn_version()
        self.type = get_sklearn_type(model)

    def save(self, file_adr):
        """
        Save model in a file.

        :param file_adr: file address
        :type file_adr: str
        :return: None
        """
        with open(file_adr, 'w') as fp:
            fp.write(self.to_json())

    def to_json(self):
        """
        Return a json-like representation of model.

        :return: model's representation as str
        """
        try:
            return json.dumps(
                {
                    "data": self.data,
                    "sklearn_version": self.version,
                    "pymilo_version": PYMILO_VERSION,
                    "model_type": self.type
                },
                indent=4
            )
        except Exception as e:
            raise PymiloSerializationException(
                {
                    'error_type': SerilaizatoinErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': {
                        "data": self.data,
                        "sklearn_version": self.version,
                        "pymilo_version": PYMILO_VERSION,
                        "model_type": self.type},
                })


class Import:
    """
    The Pymilo Import class facilitates importing of serialized models from either a designated file path or a JSON string dump.

    >>> imported_model = Import(exported_model_serialized_path)
    >>> imported_sklearn_model = imported_model.to_model()
    >>> imported_sklearn_model.predict(x_test)
    """

    def __init__(self, file_adr, json_dump=None):
        """
        Initialize the Pymilo Import instance.

        :param file_adr: the file path where the serialized model's JSON file is located.
        :type file_adr: string
        :param json_dump: the json dump of the associated model, it can be None(reading from the file_adr)
        :type json_dump: str or None
        :return: an instance of the Pymilo Import class
        """
        serialized_model_obj = None
        try:
            if json_dump and isinstance(json_dump, str):
                serialized_model_obj = json.loads(json_dump)
            else:
                with open(file_adr, 'r') as fp:
                    serialized_model_obj = json.load(fp)
            self.data = serialized_model_obj["data"]
            self.version = serialized_model_obj["sklearn_version"]
            self.type = serialized_model_obj["model_type"]
        except Exception as e:
            json_content = None
            if json_dump and isinstance(json_dump, str):
                json_content = json_dump
            else:
                with open(file_adr) as f:
                    json_content = f.readlines()
            raise PymiloDeserializationException(
                {
                    'json_file': json_content,
                    'error_type': DeSerilaizatoinErrorTypes.CORRUPTED_JSON_FILE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': serialized_model_obj})

    def to_model(self):
        """
        Convert imported model to sklearn model.

        :return: sklearn model
        """
        return to_sklearn_model(self)
