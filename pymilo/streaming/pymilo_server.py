from ..pymilo_obj import Export, Import
from .compressor import DummyCompressor
from .encryptor import DummyEncryptor
from .communicator import RESTServerCommunicator
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class PymiloServer:
    """
    The Pymilo PymiloServer class facilitates streaming the ML models.
    """

    def __init__(self):
        """
        Initialize the Pymilo PymiloServer instance.

        :return: an instance of the Pymilo PymiloServer class
        """
        self._model = None
        self._compressor = DummyCompressor()
        self._encryptor = DummyEncryptor()
        self._communicator = RESTServerCommunicator(ps=self)
        self._communicator.run()

    def export_model(self):
        """
        Export the ML model to string json dump using PyMilo Export class.

        :return: str
        """
        return Export(self._model).to_json()

    def update_model(self, serialized_model):
        """
        Update the PyMilo Server's ML model

        :param serialized_model: the json dump of a pymilo export ml model
        :type serialized_model: str
        :return: None
        """
        self._model = Import(file_adr=None, json_dump=serialized_model).to_model()

    def execute_model(self, request):
        """
        Execute the request attribute call from PyMilo Client

        :param request: request obj containing requested attribute to call with the associated args and kwargs
        :type request: obj
        :return: str | dict
        """
        gdst = GeneralDataStructureTransporter()
        attribute = request.attribute
        retrieved_attribute = getattr(self._model, attribute, None)
        if retrieved_attribute is None:
            raise Exception("The requested attribute doesn't exist in this model.")
        arguments = {
            'args': request.args,
            'kwargs': request.kwargs
        }
        args = gdst.deserialize(arguments, 'args', None)
        kwargs = gdst.deserialize(arguments, 'kwargs', None)
        output = retrieved_attribute(*args, **kwargs)
        if isinstance(output, type(self._model)):
            self._model = output
            return None
        return gdst.serialize({'output': output}, 'output', None)
