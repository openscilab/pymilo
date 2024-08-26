import numpy as np
from sklearn.metrics import mean_squared_error
from pymilo.streaming.compressor import Compression
from pymilo.streaming.pymilo_client import PymiloClient, Mode
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario3(compression_method):
    # [PyMilo Server is initialized with ML Model]
    # 1. calculate mse in server
    # 2. download model in local
    # 3. calculate mse in local
    # 4. compare results

    # 1.
    _, _, x_test, y_test = prepare_simple_regression_datasets()
    client = PymiloClient(
        mode=Mode.LOCAL,
        compressor=Compression[compression_method],
        server_url="http://127.0.0.1:9000",
        )
    client.toggle_mode(Mode.DELEGATE)
    result = client.predict(x_test)
    mse_server = mean_squared_error(y_test, result)

    # 2.
    client.download()

    # 3.
    client.toggle_mode(mode=Mode.LOCAL)
    result = client.predict(x_test)
    mse_local = mean_squared_error(y_test, result)

    # 4.
    return np.abs(mse_server-mse_local)
