import os
import json
import time
import pytest
import subprocess
from filecmp import cmp
from sys import executable
from scenarios.scenario1 import scenario1
from scenarios.scenario2 import scenario2
from scenarios.scenario3 import scenario3
from scenarios.scenario4 import scenario4
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
    yield (compression_method, "REST")
    server_proc.terminate()


@pytest.fixture(
    scope="session",
    params=["REST", "WEBSOCKET"])
def prepare_ml_server(request):
    communication_protocol = request.param
    compression_method = "ZLIB"
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


@pytest.fixture(
    scope="function",
    params=["REST", "WEBSOCKET"])
def prepare_access_control_server(request):
    communication_protocol = request.param
    compression_method = "ZLIB"
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
            "--port", "8500",
        ],
        )
    time.sleep(10)
    yield (compression_method, communication_protocol)
    server_proc.terminate()
    time.sleep(2)


@pytest.fixture(scope="session")
def prepare_cli_server():
    # The PyMilo CLI is what the generated Dockerfiles run, so it needs coverage of its own.
    compression_method, communication_protocol = "ZLIB", "REST"
    server_proc = subprocess.Popen(
        [
            executable,
            "-m", "pymilo",
            "--compression", compression_method,
            "--protocol", communication_protocol,
            "--port", "8600",
            "--load", os.path.join(
                os.getcwd(), "tests", "test_exceptions", "valid_jsons", "linear_regression.json"),
        ],
        )
    time.sleep(10)
    yield (compression_method, communication_protocol)
    server_proc.terminate()
    time.sleep(2)


def test1(prepare_bare_server):
    compression_method, communication_protocol = prepare_bare_server
    assert scenario1(compression_method, communication_protocol) == 0


def test2(prepare_bare_server):
    compression_method, communication_protocol = prepare_bare_server
    assert scenario2(compression_method, communication_protocol) == 0


def test3(prepare_ml_server):
    compression_method, communication_protocol = prepare_ml_server
    assert scenario3(compression_method, communication_protocol) == 0


def test4(prepare_access_control_server):
    compression_method, communication_protocol = prepare_access_control_server
    assert scenario4(compression_method, communication_protocol) == 0


def test5(prepare_bare_server):
    pytest.importorskip("xgboost")
    from scenarios.scenario5 import scenario5
    compression_method, communication_protocol = prepare_bare_server
    assert scenario5(compression_method, communication_protocol) == 0


def test6(prepare_cli_server):
    # same scenario as test3, against a server the PyMilo CLI preloaded through `--load`
    compression_method, communication_protocol = prepare_cli_server
    assert scenario3(compression_method, communication_protocol, server_url="127.0.0.1:8600") == 0


@pytest.mark.parametrize("cli_args, port", [(["--init", "LinearRegression"], "8400"), (["--bare"], "8401")])
def test_cli_server_boots(cli_args, port):
    server_proc = subprocess.Popen([executable, "-m", "pymilo", "--port", port] + cli_args)
    time.sleep(8)
    is_running = server_proc.poll() is None
    server_proc.terminate()
    time.sleep(2)
    assert is_running


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
    generate_dockerfile(
        dockerfile_name="Dockerfile",
        init_model="LinearRegression",
        )
    # the `--init` branch of generate_dockerfile used to emit a malformed (non-JSON) CMD
    with open('Dockerfile') as f:
        cmd_line = [line for line in f.read().splitlines() if line.startswith("CMD ")][0]
    assert json.loads(cmd_line[len("CMD "):])[-2:] == ["--init", "LinearRegression"]
    os.remove(path='Dockerfile')
    assert r1 and r2
