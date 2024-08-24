# -*- coding: utf-8 -*-
"""PyMilo RESTFull Communication Mediums."""
import json
import uvicorn
import requests
from pydantic import BaseModel
from fastapi import FastAPI, Request
from .interfaces import ClientCommunicator


class RESTClientCommunicator(ClientCommunicator):
    """Facilitate working with the communication medium from the client side for the REST protocol."""

    def __init__(self, server_url):
        """
        Initialize the Pymilo RESTClientCommunicator instance.

        :param server_url: the url to which PyMilo Server listens
        :type server_url: str
        :return: an instance of the Pymilo RESTClientCommunicator class
        """
        self._server_url = server_url
        self.session = requests.Session()
        retries = requests.adapters.Retry(
            total=10,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        self.session.mount('http://', requests.adapters.HTTPAdapter(max_retries=retries))
        self.session.mount('https://', requests.adapters.HTTPAdapter(max_retries=retries))

    def download(self, payload):
        """
        Request for the remote ML model to download.

        :param payload: download request payload
        :type payload: dict
        :return: string serialized model
        """
        response = self.session.get(url=self._server_url + "/download/", json=payload, timeout=5)
        if response.status_code != 200:
            return None
        return response.json()["payload"]

    def upload(self, payload):
        """
        Upload the local ML model to the remote server.

        :param payload: upload request payload
        :type payload: dict
        :return: True if upload was successful, False otherwise
        """
        response = self.session.post(url=self._server_url + "/upload/", json=payload, timeout=5)
        return response.status_code == 200

    def attribute_call(self, payload):
        """
        Delegate the requested attribute call to the remote server.

        :param payload: attribute call request payload
        :type payload: dict
        :return: json-encoded response of pymilo server
        """
        response = self.session.post(url=self._server_url + "/attribute_call/", json=payload, timeout=5)
        return response.json()

    def attribute_type(self, payload):
        """
        Identify the attribute type of the requested attribute.

        :param payload: attribute type request payload
        :type payload: dict
        :return: response of pymilo server
        """
        response = self.session.post(url=self._server_url + "/attribute_type/", json=payload, timeout=5)
        return response.json()


class RESTServerCommunicator():
    """Facilitate working with the communication medium from the server side for the REST protocol."""

    def __init__(
            self,
            ps,
            host: str = "127.0.0.1",
            port: int = 8000,
    ):
        """
        Initialize the Pymilo RESTServerCommunicator instance.

        :param ps: reference to the PyMilo server
        :type ps: pymilo.streaming.PymiloServer
        :param host: the url to which PyMilo Server listens
        :type host: str
        :param port: the port to which PyMilo Server listens
        :type port: int
        :return: an instance of the Pymilo RESTServerCommunicator class
        """
        self.app = FastAPI()
        self.host = host
        self.port = port
        self._ps = ps
        self.setup_routes()

    def setup_routes(self):
        """Configure endpoints to handle RESTClientCommunicator requests."""
        class StandardPayload(BaseModel):
            client_id: str
            model_id: str

        class DownloadPayload(StandardPayload):
            pass

        class UploadPayload(StandardPayload):
            model: str

        class AttributeCallPayload(StandardPayload):
            attribute: str
            args: list
            kwargs: dict

        class AttributeTypePayload(StandardPayload):
            attribute: str

        @self.app.get("/download/")
        async def download(request: Request):
            body = await request.json()
            body = self.parse(body)
            payload = DownloadPayload(**body)
            message = "/download request from client: {} for model: {}".format(payload.client_id, payload.model_id)
            return {
                "message": message,
                "payload": self._ps.export_model(),
            }

        @self.app.post("/upload/")
        async def upload(request: Request):
            body = await request.json()
            body = self.parse(body)
            payload = UploadPayload(**body)
            message = "/upload request from client: {} for model: {}".format(payload.client_id, payload.model_id)
            return {
                "message": message,
                "payload": self._ps.update_model(payload.model)
            }

        @self.app.post("/attribute_call/")
        async def attribute_call(request: Request):
            body = await request.json()
            body = self.parse(body)
            payload = AttributeCallPayload(**body)
            message = "/attribute_call request from client: {} for model: {}".format(
                payload.client_id, payload.model_id)
            result = self._ps.execute_model(payload)
            return {
                "message": message,
                "payload": result if result is not None else "The ML model has been updated in place."
            }

        @self.app.post("/attribute_type/")
        async def attribute_type(request: Request):
            body = await request.json()
            body = self.parse(body)
            payload = AttributeTypePayload(**body)
            message = "/attribute_type request from client: {} for model: {}".format(
                payload.client_id, payload.model_id)
            is_callable, field_value = self._ps.is_callable_attribute(payload)
            return {
                "message": message,
                "attribute type": "method" if is_callable else "field",
                "attribute value": "" if is_callable else field_value,
            }

    def parse(self, body):
        """
        Parse the compressed encrypted body of the request.

        :param body: request body
        :type body: str
        :return: the extracted decrypted version
        """
        return json.loads(
            self._ps._compressor.extract(
                self._ps._encryptor.decrypt(body)
            )
        )

    def run(self):
        """Run internal fastapi server."""
        uvicorn.run(self.app, host=self.host, port=self.port)
