import numpy as np
from pymilo.streaming import PymiloClient, Compression, CommunicationProtocol
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario1(compression_method, communication_protocol):
    # [PyMilo Server is not initialized with ML Model]
    # 1. create model in local
    # 2. train model in local
    # 3. calculate mse before streaming
    # 4. upload model to server
    # 5. download model to local
    # 6. calculate mse after streaming


    # 1.
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    linear_regression = LinearRegression()

    # 2.
    linear_regression.fit(x_train, y_train)
    client = PymiloClient(
        model=linear_regression,
        mode=PymiloClient.Mode.LOCAL,
        compressor=Compression[compression_method],
        communication_protocol=CommunicationProtocol[communication_protocol],
        )
    
    # 3. get client id + get ml model id [from remote server]
    client.register()
    client.register_ml_model()

    # 4.
    result = client.predict(x_test)
    mse_before = mean_squared_error(y_test, result)

    # 5.
    client.upload()
    # 6.
    client.download()

    # 7.
    result = client.predict(x_test)
    mse_after = mean_squared_error(y_test, result)
    return np.abs(mse_after-mse_before)
