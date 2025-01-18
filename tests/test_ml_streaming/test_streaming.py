import os
import time
import pytest
import subprocess
from filecmp import cmp
from sys import executable
from scenarios.scenario1 import scenario1
from scenarios.scenario2 import scenario2
from scenarios.scenario3 import scenario3
from pymilo.streaming.util import generate_dockerfile

@pytest.fixture(
    scope="session",
    params=["NULL", "GZIP", "ZLIB", "LZMA", "BZ2"])
def prepare_bare_server(request):
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
            "--compression", compression_method,
            "--protocol", "REST"
        ],
        )
    time.sleep(10)
    yield (server_proc, compression_method, "REST")
    server_proc.terminate()


@pytest.fixture(
    scope="session",
    params=["REST", "WEBSOCKET"])
def prepare_ml_server(request):
    communication_protocol = request.param
    compression_method = "ZLIB"
    print(communication_protocol)
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
            "--compression", compression_method,
            "--protocol", communication_protocol,
            "--init",
        ],
        )
    time.sleep(5)
    yield (server_proc, compression_method, communication_protocol)
    server_proc.terminate()


def test1(prepare_bare_server):
    _, compression_method, communication_protocol = prepare_bare_server
    assert scenario1(compression_method, communication_protocol) == 0


def test2(prepare_bare_server):
    _, compression_method, communication_protocol = prepare_bare_server
    assert scenario2(compression_method, communication_protocol) == 0


def test3(prepare_ml_server):
    _, compression_method, communication_protocol = prepare_ml_server
    assert scenario3(compression_method, communication_protocol) == 0


def test_dockerfile():
    docker_files_folder = os.path.join(
        os.getcwd(),
        "tests",
        "test_ml_streaming",
        "docker_files",
    )
    generate_dockerfile(
        dockerfile_name="Dockerfile1",
        model_path="https://raw.githubusercontent.com/openscilab/pymilo/main/tests/test_exceptions/valid_jsons/linear_regression.json")
    r1 = cmp('Dockerfile1', os.path.join(
        docker_files_folder,
        "Dockerfile1"
        )
    )
    generate_dockerfile(
        dockerfile_name="Dockerfile2",
        model_path="linear_regression.json",
        )
    r2 = cmp('Dockerfile2', os.path.join(
        docker_files_folder,
        "Dockerfile2"
        )
    )
    assert r1 and r2
