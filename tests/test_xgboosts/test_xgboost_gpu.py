import json
import os

import pytest
from numpy import allclose

from pymilo import Export, Import
from pymilo.pymilo_param import XGBOOST_MODEL_TABLE, NOT_SUPPORTED
from pymilo.transporters.xgboost_transporter import (
    CPU_DEVICE,
    booster_from_transparent_dict,
    extract_requested_device,
    is_gpu_device,
    is_xgboost_gpu_available,
    rewrite_gpu_fields_to_cpu,
)
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

# Never expose a GPU to these tests, even if the host has one.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

pytestmark = pytest.mark.skipif(
    not xgboost_available or XGBOOST_MODEL_TABLE.get("XGBClassifier") == NOT_SUPPORTED,
    reason="xgboost is not installed",
)


def _cpu_model_and_data():
    from xgboost import XGBClassifier
    x_train, y_train, x_test, _ = prepare_simple_classification_datasets()
    model = XGBClassifier(**cpu_estimator_kwargs(XGBClassifier)).fit(x_train, y_train)
    return model, x_test


def test_gpu_helpers_recognize_cuda_specifiers():
    assert is_gpu_device("cuda") is True
    assert is_gpu_device("cuda:0") is True
    assert is_gpu_device("gpu") is True
    assert is_gpu_device("cpu") is False
    assert is_gpu_device(None) is False


def test_rewrite_gpu_fields_to_cpu_is_recursive():
    payload = {
        "learner": {
            "generic_param": {"device": "cuda:0", "tree_method": "gpu_hist"},
            "nested": [{"device": "gpu"}],
        }
    }
    rewrite_gpu_fields_to_cpu(payload)
    assert payload["learner"]["generic_param"]["device"] == CPU_DEVICE
    assert payload["learner"]["generic_param"]["tree_method"] == "hist"
    assert payload["learner"]["nested"][0]["device"] == CPU_DEVICE


def test_exported_cpu_model_records_cpu_device_metadata():
    model, _ = _cpu_model_and_data()
    payload = json.loads(Export(model).to_json())
    booster_body = payload["data"]["_Booster"]["pymilo-xgboost-booster"]
    requested = extract_requested_device(booster_body["model-json"], booster_body["config"])
    assert requested == "cpu"
    assert booster_body["requested_device"] == "cpu"
    assert payload["data"]["device"] == "cpu"


def test_simulated_gpu_payload_falls_back_to_cpu_on_import():
    """Simulate a GPU-trained JSON file without touching a GPU.

    The serialized CPU model is rewritten so its metadata claims ``device=cuda``.
    Import must remap that request onto CPU and keep predictions identical.
    """
    model, x_test = _cpu_model_and_data()
    payload = json.loads(Export(model).to_json())
    booster_body = payload["data"]["_Booster"]["pymilo-xgboost-booster"]
    booster_body["requested_device"] = "cuda"
    if isinstance(booster_body.get("config"), dict):
        generic = booster_body["config"].setdefault("learner", {}).setdefault("generic_param", {})
        generic["device"] = "cuda"
    payload["data"]["device"] = "cuda"

    imported = Import(json_dump=json.dumps(payload)).to_model()
    assert allclose(imported.predict_proba(x_test), model.predict_proba(x_test), rtol=1e-5, atol=1e-6)
    rebuilt = booster_from_transparent_dict(booster_body, map_gpu_to_cpu=True)
    assert rebuilt.num_features() == model.get_booster().num_features()


def test_gpu_availability_probe_does_not_allocate_a_device():
    # The helper only reads build_info(); it must not raise and must not train.
    available = is_xgboost_gpu_available()
    assert available in (True, False)


@pytest.mark.skipif(True, reason="Real CUDA execution is disabled; GPU support is simulated in this suite.")
def test_real_gpu_training_disabled():
    """Placeholder for a real GPU training path.

    Kept so the repository documents a GPU test entry point. The test is
    unconditionally skipped: training on GPU is not exercised here.
    """
    from xgboost import XGBClassifier
    x_train, y_train, _, _ = prepare_simple_classification_datasets()
    XGBClassifier(device="cuda", n_estimators=2, max_depth=1, verbosity=0).fit(x_train, y_train)
