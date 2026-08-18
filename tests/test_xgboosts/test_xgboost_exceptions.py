import pytest

from pymilo import Export
from pymilo.chains.xgboost_chain import xgboost_chain
from pymilo.exceptions.serialize_exception import PymiloSerializationException
from pymilo.pymilo_param import XGBOOST_MODEL_TABLE, NOT_SUPPORTED
from pymilo.transporters.transporter import Command
from pymilo.transporters.xgboost_transporter import booster_from_transparent_dict
from xgboost_test_helpers import xgboost_available

pytestmark = pytest.mark.skipif(
    not xgboost_available or XGBOOST_MODEL_TABLE.get("XGBClassifier") == NOT_SUPPORTED,
    reason="xgboost is not installed",
)


def test_irrelevant_model_rejected_by_xgboost_chain():
    from sklearn.linear_model import LinearRegression
    from pymilo.utils.data_exporter import prepare_simple_regression_datasets
    x_train, y_train, _, _ = prepare_simple_regression_datasets()
    linear = LinearRegression().fit(x_train, y_train)
    with pytest.raises(PymiloSerializationException):
        xgboost_chain.transport(linear, Command.SERIALIZE)


def test_missing_model_json_raises():
    with pytest.raises(ValueError):
        booster_from_transparent_dict({"config": {}})


def test_invalid_payload_type_raises():
    with pytest.raises(TypeError):
        booster_from_transparent_dict("not-a-dict")


def test_export_invalid_object_is_not_an_xgboost_model():
    class NotAModel(object):
        def fit(self, x, y):
            return self

    with pytest.raises(Exception):
        Export(NotAModel())
