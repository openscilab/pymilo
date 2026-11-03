import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRegressor

from pymilo.streaming import PymiloClient, Compression, CommunicationProtocol
from sklearn.metrics import mean_squared_error
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario5(compression_method, communication_protocol):
    # XGBoost model through ML streaming (same flow as scenario1):
    # 1. create + train XGBRegressor locally
    # 2. register on server
    # 3. measure mse before streaming
    # 4. upload / download
    # 5. measure mse after streaming — must match

    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    model = XGBRegressor(
        n_estimators=8,
        max_depth=3,
        random_state=0,
        verbosity=0,
    )
    model.fit(x_train, y_train)
    client = PymiloClient(
        model=model,
        mode=PymiloClient.Mode.LOCAL,
        compressor=Compression[compression_method],
        communication_protocol=CommunicationProtocol[communication_protocol],
    )

    client.register()
    client.register_ml_model()

    result = client.predict(x_test)
    mse_before = mean_squared_error(y_test, result)

    client.upload()
    client.download()

    result = client.predict(x_test)
    mse_after = mean_squared_error(y_test, result)

    return np.abs(mse_after - mse_before)
