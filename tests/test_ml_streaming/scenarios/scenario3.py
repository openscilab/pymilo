import numpy as np
from sklearn.metrics import mean_squared_error
from pymilo.streaming import PymiloClient, Compression
from pymilo.streaming.communicator import ClientCommunicationProtocol
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario3(compression_method, communication_protocol):
    # [PyMilo Server is initialized with ML Model]
    # 1. calculate mse in server
    # 2. download model in local
    # 3. calculate mse in local
    # 4. compare results

    # 1.
    _, _, x_test, y_test = prepare_simple_regression_datasets()
    client = PymiloClient(
        mode=PymiloClient.Mode.LOCAL,
        compressor=Compression[compression_method],
        server_url="127.0.0.1:9000",
        client_communicator=ClientCommunicationProtocol[communication_protocol],
        )
    client.toggle_mode(PymiloClient.Mode.DELEGATE)
    result = client.predict(x_test)
    mse_server = mean_squared_error(y_test, result)

    # 2.
    client.download()

    # 3.
    client.toggle_mode(mode=PymiloClient.Mode.LOCAL)
    result = client.predict(x_test)
    mse_local = mean_squared_error(y_test, result)

    # 4.
    return np.abs(mse_server-mse_local)
