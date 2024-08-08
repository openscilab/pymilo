import os
import time
import pytest
import subprocess
from sys import executable
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
            executable,
            path,
        ],
        )
    time.sleep(2)
    yield server_proc
    server_proc.terminate()

def test1(prepare_server):
    assert scenario1() == 0

def test2(prepare_server):
    assert scenario2() == 0
