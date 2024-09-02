import numpy as np
from pymilo.streaming import Compression
from pymilo.streaming import PymiloClient
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario2(compression_method):
    # [PyMilo Server is not initialized with ML Model]
    # 1. create model in local
    # 2. upload model to server
    # 3. train model in server
    # 4. calculate mse in server
    # 5. download model to local
    # 6. calculate mse in local


    # 1.
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    linear_regression = LinearRegression()
    client = PymiloClient(model=linear_regression, mode=PymiloClient.Mode.LOCAL, compressor=Compression[compression_method])

    # 2.
    client.upload()

    # 3.
    client.toggle_mode(PymiloClient.Mode.DELEGATE)
    client.fit(x_train, y_train)
    remote_field = client.coef_

    # 4.
    result = client.predict(x_test)
    mse_server = mean_squared_error(y_test, result)

    # 5.
    client.download()

    # 6.
    client.toggle_mode(mode=PymiloClient.Mode.LOCAL)
    local_field = client.coef_
    result = client.predict(x_test)
    mse_local = mean_squared_error(y_test, result)
    return np.abs(mse_server-mse_local) + np.abs(np.sum(local_field-remote_field))
