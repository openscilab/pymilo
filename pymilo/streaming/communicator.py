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
        is_valid, server_url = validate_http_url(server_url)
        if not is_valid:
            raise Exception(PYMILO_INVALID_URL)
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
        self._ps = ps
        self.host = host
        self.port = port
        self.app = FastAPI()
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


class WebSocketClientCommunicator:
    """Facilitate working with the communication medium from the client side for the WebSocket protocol."""

    def __init__(
            self,
            server_url: str = "ws://127.0.0.1:8000"
            ):
        """
        Initialize the WebSocketClientCommunicator instance.

        :param server_url: The WebSocket server URL to connect to.
        :type server_url: str
        :return: an instance of the Pymilo WebSocketClientCommunicator class
        """
        is_valid, url = validate_websocket_url(server_url)
        if not is_valid:
            raise Exception(PYMILO_INVALID_URL)
        self.server_url = url
        self.websocket = None
        self.connection_established = asyncio.Event()  # Event to signal connection status

        # Create the event loop in a background thread
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop_in_thread, daemon=True)
        self.thread.start()

        # Connect asynchronously in the background thread
        self.loop.call_soon_threadsafe(self.loop.create_task, self.connect())

    def _run_loop_in_thread(self):
        """
        Run the event loop in a separate thread.

        This allows non-blocking execution of asynchronous functions.
        """
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    async def connect(self):
        """Establish a WebSocket connection with the server."""
        if self.websocket is None or self.websocket.closed:
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

        :param action: The type of action to perform (e.g., 'download', 'upload').
        :type action: str
        :param payload: The payload associated with the action.
        :type payload: dict
        :return: The server's response as a JSON object.
        """
        await self.connection_established.wait()

        if self.websocket is None or self.websocket.closed:
            raise RuntimeError(PYMILO_CLIENT_WEBSOCKET_NOT_CONNECTED)

        message = json.dumps({"action": action, "payload": payload})
        await self.websocket.send(message)
        response = await self.websocket.recv()
        return json.loads(response)

    def download(self, payload: dict) -> dict:
        """
        Request the remote ML model to download.

        :param payload: The payload for the download request.
        :type payload: dict
        :return: The downloaded model data.
        """
        response_future = asyncio.run_coroutine_threadsafe(
            self.send_message("download", payload), self.loop
        )
        response = response_future.result()
        return response.get("payload")

    def upload(self, payload: dict) -> bool:
        """
        Upload the local ML model to the remote server.

        :param payload: The payload for the upload request.
        :type payload: dict
        :return: True if the upload request is acknowledged.
        """
        response_future = asyncio.run_coroutine_threadsafe(
            self.send_message("upload", payload), self.loop
        )
        response = response_future.result()
        return response.get("message") == "Upload request received."

    def attribute_call(self, payload: dict) -> dict:
        """
        Delegate the requested attribute call to the remote server.

        :param payload: The payload containing attribute call details.
        :type payload: dict
        :return: The server's response to the attribute call.
        """
        response_future = asyncio.run_coroutine_threadsafe(
            self.send_message("attribute_call", payload), self.loop
        )
        response = response_future.result()
        return response

    def attribute_type(self, payload: dict) -> dict:
        """
        Identify the attribute type of the requested attribute.

        :param payload: The payload containing attribute type request.
        :type payload: dict
        :return: The server's response with the attribute type.
        """
        response_future = asyncio.run_coroutine_threadsafe(
            self.send_message("attribute_type", payload), self.loop
        )
        response = response_future.result()
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

        :param ps: Reference to the PyMilo server.
        :type ps: pymilo.streaming.PymiloServer
        :param host: The WebSocket server host address.
        :type host: str
        :param port: The WebSocket server port.
        :type port: int
        :return: An instance of the WebSocketServerCommunicator class.
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

        :param websocket: The WebSocket connection to accept.
        :type websocket: WebSocket
        """
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        """
        Handle WebSocket disconnection.

        :param websocket: The WebSocket connection to remove.
        :type websocket: WebSocket
        """
        self.active_connections.remove(websocket)

    async def handle_message(self, websocket: WebSocket, message: str):
        """
        Handle messages received from WebSocket clients.

        :param websocket: The WebSocket connection from which the message was received.
        :type websocket: WebSocket
        :param message: The message received from the client.
        :type message: str
        """
        try:
            message = json.loads(message)
            action = message['action']
            print(f"Server received action: {action}")
            payload = self.parse(message['payload'])

            if action == "download":
                response = self._handle_download()
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

    def _handle_download(self) -> dict:
        """
        Handle download requests.

        :return: A response containing the exported model.
        """
        return {
            "message": "Download request received.",
            "payload": self._ps.export_model(),
        }

    def _handle_upload(self, payload: dict) -> dict:
        """
        Handle upload requests.

        :param payload: The payload containing the model data to upload.
        :type payload: dict
        :return: A response indicating that the upload was processed.
        """
        return {
            "message": "Upload request received.",
            "payload": self._ps.update_model(payload["model"]),
        }

    def _handle_attribute_call(self, payload: dict) -> dict:
        """
        Handle attribute call requests.

        :param payload: The payload containing the attribute call details.
        :type payload: dict
        :return: A response with the result of the attribute call.
        """
        result = self._ps.execute_model(payload)
        return {
            "message": "Attribute call executed.",
            "payload": result if result else "The ML model has been updated in place.",
        }

    def _handle_attribute_type(self, payload: dict) -> dict:
        """
        Handle attribute type queries.

        :param payload: The payload containing the attribute to query.
        :type payload: dict
        :return: A response with the attribute type and value.
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

        :param message: The encrypted and compressed message to parse.
        :type message: str
        :return: The decrypted and extracted version of the message.
        """
        return json.loads(
            self._ps._compressor.extract(
                self._ps._encryptor.decrypt(message)
            )
        )

    def run(self):
        """Run the internal FastAPI server."""
        uvicorn.run(self.app, host=self.host, port=self.port)


