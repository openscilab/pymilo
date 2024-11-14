# -*- coding: utf-8 -*-
"""PyMilo modules."""
import os
import re
import json
from copy import deepcopy
from warnings import warn
from traceback import format_exc
from .utils.util import get_sklearn_type, download_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from .pymilo_func import get_sklearn_data, get_sklearn_version, to_sklearn_model
from .exceptions.serialize_exception import PymiloSerializationException, SerializationErrorTypes
from .exceptions.deserialize_exception import PymiloDeserializationException, DeserializationErrorTypes
from .pymilo_param import PYMILO_VERSION, UNEQUAL_PYMILO_VERSIONS, UNEQUAL_SKLEARN_VERSIONS
from .pymilo_param import INVALID_IMPORT_INIT_PARAMS, BATCH_IMPORT_INVALID_DIRECTORY


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
        self.data = get_sklearn_data(deepcopy(model))
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
                    'error_type': SerializationErrorTypes.VALID_MODEL_INVALID_INTERNAL_STRUCTURE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': {
                        "data": self.data,
                        "sklearn_version": self.version,
                        "pymilo_version": PYMILO_VERSION,
                        "model_type": self.type},
                })

    @staticmethod
    def batch_export(models, file_addr, run_parallel=False):
        """
        Export a batch of models to individual JSON files in a specified directory.

        This method takes a list of trained models and exports each one into a JSON file. The models
        are exported concurrently using multiple threads, where each model is saved to a file named
        'model_{index}.json' in the provided directory.

        :param models: list of models to get exported.
        :type models: list
        :param file_addr: the directory where exported JSON files will be saved.
        :type file_addr: str
        :param run_parallel: flag indicating the parallel execution of exports
        :type run_parallel: boolean
        :return: the count of models exported successfully
        """
        if not os.path.exists(file_addr):
            os.mkdir(file_addr)

        def export_model(model, index):
            try:
                Export(model).save(file_adr=os.path.join(file_addr, f"model_{index}.json"))
                return 1
            except Exception as _:
                return 0
        if run_parallel:
            with ThreadPoolExecutor() as executor:
                futures = [executor.submit(export_model, model, index) for index, model in enumerate(models)]
                count = 0
                for future in as_completed(futures):
                    count += future.result()
                return count
        else:
            count = 0
            for index, model in enumerate(models):
                count += export_model(model, index)
            return count


class Import:
    """
    The Pymilo Import class facilitates importing of serialized models from either a designated file path or a JSON string dump.

    >>> imported_model = Import(exported_model_serialized_path)
    >>> imported_sklearn_model = imported_model.to_model()
    >>> imported_sklearn_model.predict(x_test)
    """

    def __init__(self, file_adr=None, json_dump=None, url=None):
        """
        Initialize the Pymilo Import instance.

        :param file_adr: the file path where the serialized model's JSON file is located.
        :type file_adr: str or None
        :param json_dump: the json dump of the associated model, it can be None(reading from the file_adr)
        :type json_dump: str or None
        :param url: url to exported JSON file
        :type: str or None
        :return: an instance of the Pymilo Import class
        """
        serialized_model_obj = None
        if url is not None:
            serialized_model_obj = download_model(url)
        elif json_dump is not None and isinstance(json_dump, str):
            serialized_model_obj = json.loads(json_dump)
        elif file_adr is not None:
            with open(file_adr, 'r') as fp:
                serialized_model_obj = json.load(fp)
        else:
            raise Exception(INVALID_IMPORT_INIT_PARAMS)
        try:
            if not serialized_model_obj["pymilo_version"] == PYMILO_VERSION:
                warn(UNEQUAL_PYMILO_VERSIONS, category=Warning)
            if not serialized_model_obj["sklearn_version"] == get_sklearn_version():
                warn(UNEQUAL_SKLEARN_VERSIONS, category=Warning)
            self.data = serialized_model_obj["data"]
            self.version = serialized_model_obj["sklearn_version"]
            self.type = serialized_model_obj["model_type"]
        except Exception as e:
            json_content = None
            if json_dump and isinstance(json_dump, str):
                json_content = json_dump
            elif file_adr is not None:
                with open(file_adr) as f:
                    json_content = f.readlines()
            else:
                json_content = serialized_model_obj
            raise PymiloDeserializationException(
                {
                    'json_file': json_content,
                    'error_type': DeserializationErrorTypes.CORRUPTED_JSON_FILE,
                    'error': {
                        'Exception': repr(e),
                        'Traceback': format_exc()},
                    'object': ""})

    def to_model(self):
        """
        Convert imported model to sklearn model.

        :return: sklearn model
        """
        return to_sklearn_model(self)

    @staticmethod
    def batch_import(file_addr, run_parallel=False):
        """
        Import a batch of models from individual JSON files in a specified directory.

        This method takes a directory containing JSON files and imports each one into a model.
        The models are imported concurrently using multiple threads, ensuring that the files are
        processed in the order determined by their numeric suffixes. The function returns the
        successfully imported models in the same order as their filenames.

        :param file_addr: the directory where the JSON files to be imported are located.
        :type file_addr: str
        :param run_parallel: flag indicating the parallel execution of imports
        :type run_parallel: boolean
        :return: a tuple containing the count of models imported successfully and a list of the
                imported models in their filename order.
        """
        if not os.path.exists(file_addr):
            raise FileNotFoundError(BATCH_IMPORT_INVALID_DIRECTORY)

        json_files = [f for f in os.listdir(file_addr) if f.endswith('.json')]
        json_files.sort(key=lambda x: int(re.search(r'_(\d+)\.json$', x).group(1)))

        models = [None] * len(json_files)
        count = 0

        def import_model(file_path, index):
            try:
                model = Import(file_path).to_model()
                return index, model
            except Exception as _:
                return index, None

        if run_parallel:
            with ThreadPoolExecutor() as executor:
                futures = {
                    executor.submit(
                        import_model,
                        os.path.join(
                            file_addr,
                            file),
                        index): index for index,
                    file in enumerate(json_files)}
                for future in as_completed(futures):
                    index, model = future.result()
                    if model is not None:
                        models[index] = model
                        count += 1
                return count, models
        else:
            count = 0
            for index, file in enumerate(json_files):
                model = Import(os.path.join(file_addr, file)).to_model()
                if model is not None:
                    models[index] = model
                    count += 1
            return count, models
