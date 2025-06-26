# -*- coding: utf-8 -*-
"""PyMilo Communication Mediums."""
import uuid
import json
import asyncio
import uvicorn
import requests
import websockets
from enum import Enum
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from .interfaces import ClientCommunicator
from .param import PYMILO_INVALID_URL, PYMILO_CLIENT_WEBSOCKET_NOT_CONNECTED, REST_API_PREFIX
from .util import validate_websocket_url, validate_http_url


class RESTClientCommunicator(ClientCommunicator):
    """Facilitate working with the communication medium from the client side for the REST protocol."""

    def __init__(self, server_url):
        """
        Initialize the Pymilo RESTClientCommunicator instance.

        :param server_url: the url to which PyMilo Server listens
        :type server_url: str
        :return: an instance of the Pymilo RESTClientCommunicator class
        """
        is_valid, server_url = validate_http_url(server_url)
        if not is_valid:
            raise Exception(PYMILO_INVALID_URL)
        self._server_url = server_url.rstrip("/") + "/api/v1"
        self.session = requests.Session()
        retries = requests.adapters.Retry(
            total=10,
            backoff_factor=0.1,
            status_forcelist=[500, 502, 503, 504]
        )
        adapter = requests.adapters.HTTPAdapter(max_retries=retries)
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    def download(self, client_id, model_id):
        """
        Request for the remote ML model to download.

        :param client_id: ID of the requesting client
        :param model_id: ID of the model to download
        :return: string serialized model
        """
        url = f"{self._server_url}/clients/{client_id}/models/{model_id}/download"
        response = self.session.get(url, timeout=5)
        response.raise_for_status()
        return response.json()["payload"]

    def upload(self, client_id, model_id, model):
        """
        Upload the local ML model to the remote server.

        :param client_id: ID of the client
        :param model_id: ID of the model
        :param model: serialized model content
        :return: True if upload was successful, False otherwise
        """
        url = f"{self._server_url}/clients/{client_id}/models/{model_id}/upload"
        response = self.session.post(url, json=model, timeout=5)
        return response.status_code == 200

    def attribute_call(self, client_id, model_id, call_payload):
        """
        Delegate the requested attribute call to the remote server.

        :param client_id: ID of the client
        :param model_id: ID of the model
        :param call_payload: payload containing attribute name, args, and kwargs
        :return: json-encoded response of pymilo server
        """
        url = f"{self._server_url}/clients/{client_id}/models/{model_id}/attribute-call"
        response = self.session.post(url, json=call_payload, timeout=5)
        response.raise_for_status()
        return response.json()

    def attribute_type(self, client_id, model_id, type_payload):
        """
        Identify the attribute type of the requested attribute.

        :param client_id: ID of the client
        :param model_id: ID of the model
        :param type_payload: payload containing attribute data to inspect
        :return: response of pymilo server
        """
        url = f"{self._server_url}/clients/{client_id}/models/{model_id}/attribute-type"
        response = self.session.post(url, json=type_payload, timeout=5)
        response.raise_for_status()
        return response.json()

    def register_client(self):
        """
        Register client in the PyMiloServer.

        :return: newly allocated client id
        """
        response = self.session.get(f"{self._server_url}/clients/register", timeout=5)
        response.raise_for_status()
        return response.json()["client_id"]

    def remove_client(self, client_id):
        """
        Remove client from the PyMiloServer.

        :param client_id: id of the client to remove
        :type client_id: str
        :return: True if removal was successful, False otherwise
        """
        response = self.session.delete(f"{self._server_url}/clients/{client_id}", timeout=5)
        return response.status_code == 200

    def register_model(self, client_id):
        """
        Register ML model in the PyMiloServer.

        :param client_id: id of the client who owns the model
        :type client_id: str
        :return: newly allocated ml model id
        """
        response = self.session.post(f"{self._server_url}/clients/{client_id}/models/register", timeout=5)
        response.raise_for_status()
        return response.json()["ml_model_id"]

    def remove_model(self, client_id, model_id):
        """
        Remove ML model from the PyMiloServer.

        :param client_id: client owning the model
        :type client_id: str
        :param model_id: model to remove
        :type model_id: str
        :return: True if removal was successful, False otherwise
        """
        response = self.session.delete(f"{self._server_url}/clients/{client_id}/models/{model_id}", timeout=5)
        return response.status_code == 200

    def get_ml_models(self, client_id):
        """
        Get all ML models registered for this specific client in the PyMiloServer.

        :param client_id: client whose models are being queried
        :type client_id: str
        :return: list of ml model ids
        """
        response = self.session.get(f"{self._server_url}/clients/{client_id}/models", timeout=5)
        response.raise_for_status()
        return response.json()["ml_models_id"]

    def grant_access(self, allower_id, allowee_id, model_id):
        """
        Grant access to a model to another client.

        :param allower_id: ID of the client granting access
        :param allowee_id: ID of the client being granted access
        :param model_id: ID of the model being shared
        :return: True if successful, False otherwise
        """
        url = f"{self._server_url}/clients/{allower_id}/grant/{allowee_id}/models/{model_id}"
        response = self.session.post(url, timeout=5)
        return response.status_code == 200

    def revoke_access(self, revoker_id, revokee_id, model_id):
        """
        Revoke previously granted model access.

        :param revoker_id: ID of the client revoking access
        :param revokee_id: ID of the client whose access is being revoked
        :param model_id: ID of the model
        :return: True if successful, False otherwise
        """
        url = f"{self._server_url}/clients/{revoker_id}/revoke/{revokee_id}/models/{model_id}"
        response = self.session.post(url, timeout=5)
        return response.status_code == 200

    def get_allowance(self, allower_id):
        """
        Get the list of all allowees and their allowed models from a given allower.

        :param allower_id: ID of the allower
        :return: dict of allowees to model lists
        """
        response = self.session.get(f"{self._server_url}/clients/{allower_id}/allowances", timeout=5)
        response.raise_for_status()
        return response.json()["allowance"]

    def get_allowed_models(self, allower_id, allowee_id):
        """
        Get the list of models that one client is allowed to access from another.

        :param allower_id: ID of the model owner
        :param allowee_id: ID of the requesting client
        :return: list of model IDs
        """
        url = f"{self._server_url}/clients/{allower_id}/allowances/{allowee_id}"
        response = self.session.get(url, timeout=5)
        response.raise_for_status()
        return response.json()["allowed_models"]


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
        self._ps = ps
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.setup_routes()

    def setup_routes(self):
        """Configure endpoints to handle RESTClientCommunicator requests."""

        @self.app.get(f"{REST_API_PREFIX}/health")
        async def health():
            return {"status": "ok"}

        @self.app.get(f"{REST_API_PREFIX}/clients/register")
        async def request_client_id():
            client_id = str(uuid.uuid4())
            self._ps.init_client(client_id)
            return {"client_id": client_id}

        @self.app.delete(f"{REST_API_PREFIX}/clients/{{client_id}}")
        async def remove_client(client_id: str):
            is_succeed, detail_message = self._ps.remove_client(client_id)
            if not is_succeed:
                raise HTTPException(status_code=404, detail=detail_message)
            return {"client_id": client_id}

        @self.app.post(f"{REST_API_PREFIX}/clients/{{client_id}}/models/register")
        async def request_model(client_id: str):
            model_id = str(uuid.uuid4())
            is_succeed, detail_message = self._ps.init_ml_model(client_id, model_id)
            if not is_succeed:
                raise HTTPException(status_code=404, detail=detail_message)
            return {"client_id": client_id, "ml_model_id": model_id}

        @self.app.delete(f"{REST_API_PREFIX}/clients/{{client_id}}/models/{{ml_model_id}}")
        async def remove_model(client_id: str, ml_model_id: str):
            is_succeed, detail_message = self._ps.remove_ml_model(client_id, ml_model_id)
            if not is_succeed:
                raise HTTPException(status_code=404, detail=detail_message)
            return {"client_id": client_id, "ml_model_id": ml_model_id}

        @self.app.get(f"{REST_API_PREFIX}/clients/{{client_id}}/models")
        async def get_client_models(client_id: str):
            return {"client_id": client_id, "ml_models_id": self._ps.get_ml_models(client_id)}

        @self.app.post(f"{REST_API_PREFIX}/clients/{{allower_id}}/grant/{{allowee_id}}/models/{{model_id}}")
        async def grant_model_access(allower_id: str, allowee_id: str, model_id: str):
            is_succeed, detail_message = self._ps.grant_access(allower_id, allowee_id, model_id)
            if not is_succeed:
                raise HTTPException(status_code=404, detail=detail_message)
            return {
                "allower_id": allower_id,
                "allowee_id": allowee_id,
                "allowed_model_id": model_id
            }

        @self.app.post(f"{REST_API_PREFIX}/clients/{{revoker_id}}/revoke/{{revokee_id}}/models/{{model_id}}")
        async def revoke_model_access(revoker_id: str, revokee_id: str, model_id: str):
            is_succeed, detail_message = self._ps.revoke_access(revoker_id, revokee_id, model_id)
            if not is_succeed:
                raise HTTPException(status_code=404, detail=detail_message)
            return {
                "revoker_id": revoker_id,
                "revokee_id": revokee_id,
                "revoked_model_id": model_id
            }

        @self.app.get(f"{REST_API_PREFIX}/clients/{{allower_id}}/allowances")
        async def get_allowance(allower_id: str):
            allowance, reason = self._ps.get_clients_allowance(allower_id)
            if not allowance:
                raise HTTPException(status_code=404, detail=reason)
            return {"allower_id": allower_id, "allowance": allowance}

        @self.app.get(f"{REST_API_PREFIX}/clients/{{allower_id}}/allowances/{{allowee_id}}")
        async def get_allowed_models(allower_id: str, allowee_id: str):
            models, reason = self._ps.get_allowed_models(allower_id, allowee_id)
            if models is None:
                raise HTTPException(status_code=404, detail=reason)
            return {"allower_id": allower_id, "allowee_id": allowee_id, "allowed_models": models}

        @self.app.get(f"{REST_API_PREFIX}/clients/{{client_id}}/models/{{ml_model_id}}/download")
        async def download_model(client_id: str, ml_model_id: str):
            is_valid, reason = self._ps._validate_id(client_id, ml_model_id)
            if not is_valid:
                raise HTTPException(status_code=404, detail=reason)
            return {
                "message": f"/download request from client: {client_id} for model: {ml_model_id}",
                "payload": self._ps.export_model(client_id, ml_model_id)
            }

        @self.app.post(f"{REST_API_PREFIX}/clients/{{client_id}}/models/{{ml_model_id}}/upload")
        async def upload_model(client_id: str, ml_model_id: str, request: Request):
            model_data = self.parse(await request.json()).get("model")
            if model_data is None:
                raise HTTPException(status_code=400, detail="Missing 'model' in request")

            is_valid, reason = self._ps._validate_id(client_id, ml_model_id)
            if not is_valid:
                raise HTTPException(status_code=404, detail=reason)

            return {
                "message": f"/upload request from client: {client_id} for model: {ml_model_id}",
                "payload": self._ps.update_model(client_id, ml_model_id, model_data)
            }

        @self.app.post(f"{REST_API_PREFIX}/clients/{{client_id}}/models/{{ml_model_id}}/attribute-call")
        async def attribute_call(client_id: str, ml_model_id: str, request: Request):
            request_payload = self.parse(await request.json())
            is_valid, reason = self._ps._validate_id(client_id, ml_model_id)
            if not is_valid:
                raise HTTPException(status_code=404, detail=reason)
            result = self._ps.execute_model(request_payload)
            return {
                "message": f"/attribute_call request from client: {client_id} for model: {ml_model_id}",
                "payload": result or "The ML model has been updated in place."
            }

        @self.app.post(f"{REST_API_PREFIX}/clients/{{client_id}}/models/{{ml_model_id}}/attribute-type")
        async def attribute_type(client_id: str, ml_model_id: str, request: Request):
            request = self.parse(await request.json())
            is_valid, reason = self._ps._validate_id(client_id, ml_model_id)
            if not is_valid:
                raise HTTPException(status_code=404, detail=reason)
            is_callable, field_value = self._ps.is_callable_attribute(request)
            return {
                "message": f"/attribute_type request from client: {client_id} for model: {ml_model_id}",
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


class WebSocketClientCommunicator(ClientCommunicator):
    """Facilitate working with the communication medium from the client side for the WebSocket protocol."""

    def __init__(
            self,
            server_url: str = "ws://127.0.0.1:8000"
    ):
        """
        Initialize the WebSocketClientCommunicator instance.

        :param server_url: the WebSocket server URL to connect to.
        :type server_url: str
        :return: an instance of the Pymilo WebSocketClientCommunicator class
        """
        is_valid, url = validate_websocket_url(server_url)
        if not is_valid:
            raise Exception(PYMILO_INVALID_URL)
        self.server_url = url
        self.websocket = None
        self.connection_established = asyncio.Event()  # Event to signal connection status
        # check for even loop existance
        if asyncio._get_running_loop() is None:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        else:
            self.loop = asyncio.get_event_loop()
        self.loop.run_until_complete(self.connect())

    def is_socket_closed(self):
        """
        Check if the WebSocket connection is closed.

        :return: `True` if the WebSocket connection is closed or uninitialized, `False` otherwise.
        """
        if self.websocket is None:
            return True
        elif hasattr(self.websocket, "closed"):  # For older versions
            return self.websocket.closed
        elif hasattr(self.websocket, "state"):  # For newer versions
            return self.websocket.state is websockets.protocol.State.CLOSED

    async def connect(self):
        """Establish a WebSocket connection with the server."""
        if self.is_socket_closed():
            self.websocket = await websockets.connect(self.server_url)
            print("Connected to the WebSocket server.")
            self.connection_established.set()

    async def disconnect(self):
        """Close the WebSocket connection."""
        if self.websocket:
            await self.websocket.close()

    async def send_message(self, action: str, payload: dict) -> dict:
        """
        Send a message to the WebSocket server.

        :param action: the type of action to perform (e.g., 'download', 'upload').
        :type action: str
        :param payload: the payload associated with the action.
        :type payload: dict
        :return: the server's response as a JSON object.
        """
        await self.connection_established.wait()

        if self.is_socket_closed():
            raise RuntimeError(PYMILO_CLIENT_WEBSOCKET_NOT_CONNECTED)

        message = json.dumps({"action": action, "payload": payload})
        await self.websocket.send(message)
        response = await self.websocket.recv()
        return json.loads(response)

    def download(self, payload: dict) -> dict:
        """
        Request the remote ML model to download.

        :param payload: the payload for the download request.
        :type payload: dict
        :return: the downloaded model data.
        """
        response = self.loop.run_until_complete(
            self.send_message("download", payload)
        )
        return response.get("payload")

    def upload(self, payload: dict) -> bool:
        """
        Upload the local ML model to the remote server.

        :param payload: the payload for the upload request.
        :type payload: dict
        :return: true if the upload request is acknowledged.
        """
        response = self.loop.run_until_complete(
            self.send_message("upload", payload)
        )
        return response.get("message") == "Upload request received."

    def attribute_call(self, payload: dict) -> dict:
        """
        Delegate the requested attribute call to the remote server.

        :param payload: the payload containing attribute call details.
        :type payload: dict
        :return: the server's response to the attribute call.
        """
        response = self.loop.run_until_complete(
            self.send_message("attribute_call", payload)
        )
        return response

    def attribute_type(self, payload: dict) -> dict:
        """
        Identify the attribute type of the requested attribute.

        :param payload: the payload containing attribute type request.
        :type payload: dict
        :return: the server's response with the attribute type.
        """
        response = self.loop.run_until_complete(
            self.send_message("attribute_type", payload)
        )
        return response


class WebSocketServerCommunicator:
    """Facilitate working with the communication medium from the server side for the WebSocket protocol."""

    def __init__(
            self,
            ps,
            host: str = "127.0.0.1",
            port: int = 8000,
    ):
        """
        Initialize the WebSocketServerCommunicator instance.

        :param ps: reference to the PyMilo server.
        :type ps: pymilo.streaming.PymiloServer
        :param host: the WebSocket server host address.
        :type host: str
        :param port: the WebSocket server port.
        :type port: int
        :return: an instance of the WebSocketServerCommunicator class.
        """
        self._ps = ps
        self.host = host
        self.port = port
        self.app = FastAPI()
        self.active_connections: list[WebSocket] = []
        self.setup_routes()

    def setup_routes(self):
        """Configure the WebSocket endpoint to handle client connections."""
        @self.app.websocket("/")
        async def websocket_endpoint(websocket: WebSocket):
            await self.connect(websocket)
            try:
                while True:
                    message = await websocket.receive_text()
                    await self.handle_message(websocket, message)
            except WebSocketDisconnect:
                self.disconnect(websocket)

    async def connect(self, websocket: WebSocket):
        """
        Accept a WebSocket connection and store it.

        :param websocket: the WebSocket connection to accept.
        :type websocket: webSocket
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        Handle WebSocket disconnection.

        :param websocket: the WebSocket connection to remove.
        :type websocket: webSocket
        """
        self.active_connections.remove(websocket)

    async def handle_message(self, websocket: WebSocket, message: str):
        """
        Handle messages received from WebSocket clients.

        :param websocket: the WebSocket connection from which the message was received.
        :type websocket: webSocket
        :param message: the message received from the client.
        :type message: str
        """
        try:
            message = json.loads(message)
            action = message['action']
            print(f"Server received action: {action}")
            payload = self.parse(message['payload'])

            if action == "download":
                response = self._handle_download(payload)
            elif action == "upload":
                response = self._handle_upload(payload)
            elif action == "attribute_call":
                response = self._handle_attribute_call(payload)
            elif action == "attribute_type":
                response = self._handle_attribute_type(payload)
            else:
                response = {"error": f"Unknown action: {action}"}

            await websocket.send_text(json.dumps(response))
        except Exception as e:
            await websocket.send_text(json.dumps({"error": str(e)}))

    def _handle_download(self, payload) -> dict:
        """
        Handle download requests.

        :param payload: the payload containing the ids associated with the requested model for download.
        :type payload: dict
        :return: a response containing the exported model.
        """
        return {
            "message": "Download request received.",
            "payload": self._ps.export_model(
                payload["client_id"],
                payload["ml_model_id"],
            ),
        }

    def _handle_upload(self, payload: dict) -> dict:
        """
        Handle upload requests.

        :param payload: the payload containing the model data to upload.
        :type payload: dict
        :return: a response indicating that the upload was processed.
        """
        return {
            "message": "Upload request received.",
            "payload": self._ps.update_model(payload["model"]),
        }

    def _handle_attribute_call(self, payload: dict) -> dict:
        """
        Handle attribute call requests.

        :param payload: the payload containing the attribute call details.
        :type payload: dict
        :return: a response with the result of the attribute call.
        """
        result = self._ps.execute_model(payload)
        return {
            "message": "Attribute call executed.",
            "payload": result if result else "The ML model has been updated in place.",
        }

    def _handle_attribute_type(self, payload: dict) -> dict:
        """
        Handle attribute type queries.

        :param payload: the payload containing the attribute to query.
        :type payload: dict
        :return: a response with the attribute type and value.
        """
        is_callable, field_value = self._ps.is_callable_attribute(payload)
        return {
            "message": "Attribute type query executed.",
            "attribute type": "method" if is_callable else "field",
            "attribute value": "" if is_callable else field_value,
        }

    def parse(self, message: str) -> dict:
        """
        Parse the encrypted and compressed message.

        :param message: the encrypted and compressed message to parse.
        :type message: str
        :return: the decrypted and extracted version of the message.
        """
        return json.loads(
            self._ps._compressor.extract(
                self._ps._encryptor.decrypt(message)
            )
        )

    def run(self):
        """Run the internal FastAPI server."""
        uvicorn.run(self.app, host=self.host, port=self.port)


class CommunicationProtocol(Enum):
    """Communication protocol."""

    REST = {
        "CLIENT": RESTClientCommunicator,
        "SERVER": RESTServerCommunicator,
    }
    WEBSOCKET = {
        "CLIENT": WebSocketClientCommunicator,
        "SERVER": WebSocketServerCommunicator,
    }
