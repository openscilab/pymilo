from .encryptor import DummyEncryptor
from .compressor import DummyCompressor
from ..pymilo_obj import Export, Import
from .communicator import RESTClientCommunicator
from ..transporters.general_data_structure_transporter import GeneralDataStructureTransporter
class PymiloClient:

    def __init__(
            self,
            model=None,
            mode="LOCAL",
            server="http://127.0.0.1",
            port= 8000
            ):
        self._client_id = "0x_client_id"
        self._model_id = "0x_model_id"
        self._model = model
        self._mode = mode
        self._compressor = DummyCompressor()
        self._encryptor = DummyEncryptor()
        self._communicator = RESTClientCommunicator(
            server_url="{}:{}".format(server, port)
        )

    def toggle_mode(self):
        if self._mode == "LOCAL":
            self._mode = "DELEGATE"
        else:
            self._mode = "LOCAL"

    def download(self):
        response = self._communicator.download({
            "client_id": self._client_id,    
            "model_id": self._model_id
        })
        if response.status_code != 200:
            print("Remote model download failed.")
        print("Remote model downloaded successfully.")
        serialized_model = response.json()["payload"]
        self._model = Import(file_adr=None, json_dump=serialized_model).to_model()
        print("Local model updated successfully.")

    def upload(self):
        response = self._communicator.upload({
            "client_id": self._client_id,    
            "model_id": self._model_id,
            "model": Export(self._model).to_json(),
        })
        if response.status_code == 200:
            print("Local model uploaded successfully.")
        else:
            print("Local model upload failed.")

    def __getattr__(self, attribute):
        if self._mode == "LOCAL":
            if attribute in dir(self._model):
                return getattr(self._model, attribute)
            else:
                raise AttributeError("This attribute doesn't exist either in PymiloClient or the inner ML model.")
        elif self._mode == "DELEGATE":
            gdst = GeneralDataStructureTransporter()
            def relayer(*args, **kwargs):
                print(f"Method '{attribute}' called with args: {args} and kwargs: {kwargs}")
                payload = {
                    "client_id": self._client_id,                   
                    "model_id": self._model_id,
                    'attribute': attribute,
                    'args': args,
                    'kwargs': kwargs,
                    }
                payload["args"] = gdst.serialize(payload, "args", None)
                payload["kwargs"] = gdst.serialize(payload, "kwargs", None)
                result = self._communicator.attribute_call(
                    self._encryptor.encrypt(
                        self._compressor.compress(
                            payload
                        )
                    )
                ).json()
                return gdst.deserialize(result, "payload", None)
            relayer.__doc__ = getattr(self._model.__class__, attribute).__doc__
            return relayer
