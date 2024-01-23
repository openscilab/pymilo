from sklearn.naive_bayes import CategoricalNB

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "CategoricalNB"

def categorical_naive_bayes():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    categorical_naive_bayes = CategoricalNB().fit(x_train, y_train)
    pymilo_classification_test(categorical_naive_bayes, MODEL_NAME, (x_test, y_test))

