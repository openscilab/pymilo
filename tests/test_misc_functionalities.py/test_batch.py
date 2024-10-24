import os
import re
import random
import numpy as np
from pymilo import Export, Import
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from pymilo.utils.data_exporter import prepare_simple_regression_datasets


def test_batch_execution():
    x_train, y_train, x_test, _ = prepare_simple_regression_datasets()
    linear_regression = LinearRegression()
    linear_regression.fit(x_train, y_train)
    pre_models = [linear_regression]*100
    exp_n = Export.batch_export(pre_models, os.getcwd())
    imp_n, post_models = Import.batch_import(os.getcwd())
    r_index = random.randint(0, len(post_models) - 1)
    pre_result = pre_models[r_index].predict(x_test)
    post_result = post_models[r_index].predict(x_test)
    mse = mean_squared_error(post_result, pre_result)
    pattern = re.compile(r'model_\d+\.json')
    for filename in os.listdir(os.getcwd()):
        if pattern.match(filename):
            file_path = os.path.join(os.getcwd(), filename)
            os.remove(file_path)
    return exp_n == imp_n and np.abs(mse) <= 10**(-8)
