from .interfaces import Communicator


class DummyCommunicator(Communicator):
    def send(string):
        return string 
    def receive(string):
        return string