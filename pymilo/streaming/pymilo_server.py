import json
from fastapi import Request
from ..pymilo_obj import Export, Import
from .compressor import DummyCompressor
from .encryptor import DummyEncryptor
from .communicator import RESTServerCommunicator
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter


class PymiloServer:

    def __init__(self):
        self._model = None
        self._compressor = DummyCompressor()
        self._encryptor = DummyEncryptor()
        self._communicator = RESTServerCommunicator(ps=self)
        @self._communicator.app.middleware("http")
        async def preparation(request: Request, call_next):
            body = await request.json()
            body = self._compressor.extract(
                self._encryptor.decrypt(
                    body
                )
            )
            request._body = json.dumps(body).encode('utf-8')
            print("Preparation Done!")
            response = await call_next(request)
            return response
        self._communicator.run()

    def export_model(self):
        return Export(self._model).to_json()

    def update_model(self, serialized_model):
        self._model = Import(file_adr=None, json_dump=serialized_model).to_model()

    def execute_model(self, request):
        gdst = GeneralDataStructureTransporter()
        attribute = request.attribute
        retrieved_attribute = getattr(self._model, attribute, None)
        if retrieved_attribute is None:
            return "ERROR"
        arguments = {
            'args': request.args,
            'kwargs': request.kwargs
        }
        args = gdst.deserialize(arguments, 'args', None)
        kwargs = gdst.deserialize(arguments, 'kwargs', None)
        output = retrieved_attribute(*args, **kwargs)
        if isinstance(output, type(self._model)):
            self._model = output
            return "The ML model has been updated in place."
        return gdst.serialize({'output': output}, 'output', None)
