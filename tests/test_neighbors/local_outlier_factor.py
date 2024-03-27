from sklearn.neighbors import LocalOutlierFactor

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "LocalOutlierFactor"

def local_outlier_factor():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    local_outlier_factor = LocalOutlierFactor(n_neighbors=2, novelty= True).fit(x_train, y_train)
    pymilo_classification_test(local_outlier_factor, MODEL_NAME, (x_test, y_test))
