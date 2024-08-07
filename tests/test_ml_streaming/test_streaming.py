import os
import time
import pytest
import subprocess
from scenarios.scenario1 import scenario1
from scenarios.scenario2 import scenario2

@pytest.fixture(scope="session")
def prepare_server():
    path = os.path.join(
        os.getcwd(),
        "tests",
        "test_ml_streaming",
        "run_server.py"
        )
    server_proc = subprocess.Popen(
        [
            "python",
            path,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    time.sleep(2)
    yield server_proc
    server_proc.terminate()

def test1(prepare_server):
    assert scenario1() == 0

def test2(prepare_server):
    assert scenario2() == 0
