import uvicorn
import requests
from fastapi import FastAPI, Request
from pydantic import BaseModel
from .interfaces import ClientCommunicator


class RESTClientCommunicator(ClientCommunicator):

    def __init__(self, server_url):
        self._server_url = server_url

    def download(self, payload):
        return requests.get(url=self._server_url + "/download/", json=payload, timeout=5)

    def upload(self, payload):
        return requests.post(url=self._server_url + "/upload/", json=payload, timeout=5)

    def attribute_call(self, payload):
        return requests.post(url=self._server_url + "/attribute_call/", json=payload, timeout=5)


class RESTServerCommunicator():

    def __init__(
            self,
            ps,
            host: str = "127.0.0.1",
            port: int = 8000,
            ):
        self.app = FastAPI()
        self.host = host
        self.port = port
        self._ps = ps
        self.setup_routes()

    def setup_routes(self):
        class StandardPayload(BaseModel):
            client_id: str
            model_id: str
        class DownloadPayload(StandardPayload):
            pass
        class UploadPayload(StandardPayload):
            model: str
        class AttributePayload(StandardPayload):
            attribute: str
            args: list
            kwargs: dict

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
            payload = AttributePayload(**body)
            message = "/attribute_call request from client: {} for model: {}".format(payload.client_id, payload.model_id)
            return {
                "message": message,
                "payload": self._ps.execute_model(payload)
            }

    def parse(self, body):
        return self._ps._compressor.extract(
            self._ps._encryptor.decrypt(
                body
            )
        )

    def run(self):
        uvicorn.run(self.app, host=self.host, port=self.port)
