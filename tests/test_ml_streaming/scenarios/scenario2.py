import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pymilo.streaming.pymilo_client import PymiloClient
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def scenario2():
    # 1. create model in local
    # 2. upload model to server
    # 3. train model in server
    # 4. calculate mse in server
    # 5. download model to local
    # 6. calculate mse in local


    # 1.
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    linear_regression = LinearRegression()
    client = PymiloClient(model=linear_regression, mode="LOCAL")

    # 2.
    client.upload()

    # 3.
    client.toggle_mode()
    client.fit(x_train, y_train)

    # 4.
    result = client.predict(x_test)
    mse_server = mean_squared_error(y_test, result)

    # 5.
    client.download()
    
    # 6.
    client.toggle_mode()
    result = client.predict(x_test)
    mse_local = mean_squared_error(y_test, result)
    assert np.abs(mse_server-mse_local) == 0