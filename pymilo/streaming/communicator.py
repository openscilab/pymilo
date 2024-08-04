from .interfaces import Communicator


class RESTClientCommunicator(Communicator):

    def __init__(self, server_url):
        self._server_url = server_url

    def download(self, payload):
        return requests.get(url=self._server_url + "/download/", json=payload)

    def upload(self, payload):
        return requests.post(url=self._server_url + "/upload/", json=payload)

    def attribute_call(self, payload):
        return requests.post(url=self._server_url + "/attribute_call/", json=payload)

