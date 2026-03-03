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
    # Using PyMilo direct CLI
    # server_proc = subprocess.Popen(
    #     [
    #         executable,
    #         "-m", "pymilo",
    #         "--compression", compression_method,
    #         "--protocol", "REST",
    #         "--port", "8000",
    #         "--bare",
    #     ],
    #     )
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
    yield (compression_method, "REST")
    server_proc.terminate()


@pytest.fixture(
    scope="session",
    params=["REST", "WEBSOCKET"])
def prepare_ml_server(request):
    communication_protocol = request.param
    compression_method = "ZLIB"
    # Using PyMilo direct CLI
    # server_proc = subprocess.Popen(
    #     [
    #         executable,
    #         "-m", "pymilo",
    #         "--compression", compression_method,
    #         "--protocol", communication_protocol,
    #         "--port", "9000",
    #         "--load", os.path.join(os.getcwd(), "tests", "test_exceptions", "valid_jsons", "linear_regression.json")
    #         # "--load", "https://raw.githubusercontent.com/openscilab/pymilo/main/tests/test_exceptions/valid_jsons/linear_regression.json",
    #     ],
    #     )
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
    time.sleep(10)
    yield (compression_method, communication_protocol)
    server_proc.terminate()


def test1(prepare_bare_server):
    compression_method, communication_protocol = prepare_bare_server
    assert scenario1(compression_method, communication_protocol) == 0


def test2(prepare_bare_server):
    compression_method, communication_protocol = prepare_bare_server
    assert scenario2(compression_method, communication_protocol) == 0


def test3(prepare_ml_server):
    compression_method, communication_protocol = prepare_ml_server
    assert scenario3(compression_method, communication_protocol) == 0


def test_dockerfile():
    docker_files_folder = os.path.join(
        os.getcwd(),
        "tests",
        "test_ml_streaming",
        "docker_files",
    )
    generate_dockerfile(
        dockerfile_name="Dockerfile",
        model_path="https://raw.githubusercontent.com/openscilab/pymilo/main/tests/test_exceptions/valid_jsons/linear_regression.json")
    r1 = cmp('Dockerfile', os.path.join(
        docker_files_folder,
        "Dockerfile1"
        )
    )
    generate_dockerfile(
        dockerfile_name="Dockerfile",
        model_path="linear_regression.json",
        )
    r2 = cmp('Dockerfile', os.path.join(
        docker_files_folder,
        "Dockerfile2"
        )
    )
    os.remove(path='Dockerfile')
    assert r1 and r2
