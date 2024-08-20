from pymilo.streaming.compressor import Compression
from pymilo.streaming.pymilo_server import PymiloServer

communicator = PymiloServer(compressor=Compression.LZMA)._communicator
communicator.run()
