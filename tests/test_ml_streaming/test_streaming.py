import os
import time
import pytest
import subprocess
from sys import executable
from scenarios.scenario1 import scenario1
from scenarios.scenario2 import scenario2

@pytest.fixture(scope="session", params=["NULL", "GZIP", "ZLIB", "LZMA", "BZ2"])
def prepare_server(request):
    compression_method = request.param
    path = os.path.join(
        os.getcwd(),
        "tests",
        "test_ml_streaming",
        "run_server.py",
        )
    server_proc = subprocess.Popen(
        [
            executable,
            path,
            "--compression", compression_method
        ],
        )
    time.sleep(2)
    yield (server_proc, compression_method)
    server_proc.terminate()

def test1(prepare_server):
    _, compression_method = prepare_server
    assert scenario1(compression_method) == 0

def test2(prepare_server):
    _, compression_method = prepare_server
    assert scenario2(compression_method) == 0
