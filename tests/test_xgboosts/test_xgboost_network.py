import json
from unittest.mock import Mock, patch

import pytest
from numpy import allclose

from pymilo import Export, Import
from pymilo.pymilo_param import (
    DOWNLOAD_MODEL_FAILED,
    INVALID_DOWNLOADED_MODEL,
    XGBOOST_MODEL_TABLE,
    NOT_SUPPORTED,
)
from pymilo.utils.data_exporter import prepare_simple_regression_datasets
from xgboost_test_helpers import cpu_estimator_kwargs, xgboost_available

pytestmark = pytest.mark.skipif(
    not xgboost_available or XGBOOST_MODEL_TABLE.get("XGBRegressor") == NOT_SUPPORTED,
    reason="xgboost is not installed",
)

FAKE_URL = "https://example.test/exported/xgb_regressor.json"


def _fitted_regressor():
    from xgboost import XGBRegressor
    x_train, y_train, x_test, _ = prepare_simple_regression_datasets()
    model = XGBRegressor(**cpu_estimator_kwargs(XGBRegressor)).fit(x_train, y_train)
    return model, x_test


def test_import_from_url_with_mocked_network():
    model, x_test = _fitted_regressor()
    payload = json.loads(Export(model).to_json())

    with patch("pymilo.pymilo_obj.download_model") as mocked_download:
        mocked_download.return_value = payload
        imported = Import(url=FAKE_URL).to_model()
        mocked_download.assert_called_once_with(FAKE_URL)

    assert allclose(imported.predict(x_test), model.predict(x_test), rtol=1e-5, atol=1e-6)


def test_import_from_url_via_requests_session_mock():
    model, x_test = _fitted_regressor()
    payload = json.loads(Export(model).to_json())
    response = Mock()
    response.status_code = 200
    response.json.return_value = payload

    session = Mock()
    session.get.return_value = response
    session.mount = Mock()

    with patch("pymilo.utils.util.requests.Session", return_value=session):
        imported = Import(url=FAKE_URL).to_model()

    session.get.assert_called()
    assert allclose(imported.predict(x_test), model.predict(x_test), rtol=1e-5, atol=1e-6)


def test_import_from_url_network_failure_is_simulated():
    with patch("pymilo.utils.util.requests.Session") as session_cls:
        session = session_cls.return_value
        session.get.side_effect = Exception("simulated network down")
        with pytest.raises(Exception) as exc_info:
            Import(url=FAKE_URL)
        assert DOWNLOAD_MODEL_FAILED in str(exc_info.value)


def test_import_from_url_invalid_json_payload():
    response = Mock()
    response.status_code = 200
    response.json.side_effect = ValueError("not json")

    session = Mock()
    session.get.return_value = response
    session.mount = Mock()

    with patch("pymilo.utils.util.requests.Session", return_value=session):
        with pytest.raises(Exception) as exc_info:
            Import(url=FAKE_URL)
        assert INVALID_DOWNLOADED_MODEL in str(exc_info.value)
