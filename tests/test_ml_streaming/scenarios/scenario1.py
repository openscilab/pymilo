import numpy as np
from pymilo.streaming import Compression
from pymilo.streaming import PymiloClient
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario1(compression_method):
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
    client = PymiloClient(model=linear_regression, mode=PymiloClient.Mode.LOCAL, compressor=Compression[compression_method])

    # 3.
    result = client.predict(x_test)
    mse_before = mean_squared_error(y_test, result)

    # 4.
    client.upload()
    # 5.
    client.download()

    # 6.
    result = client.predict(x_test)
    mse_after = mean_squared_error(y_test, result)
    return np.abs(mse_after-mse_before)
