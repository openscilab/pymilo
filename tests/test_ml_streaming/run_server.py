from pymilo.streaming.pymilo_server import PymiloServer

communicator = PymiloServer()._communicator
communicator.run()