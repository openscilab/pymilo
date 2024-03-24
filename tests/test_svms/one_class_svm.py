from sklearn.svm import OneClassSVM

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "OneClassSVM"

def one_class_svm():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    one_class_svm = OneClassSVM(gamma='auto').fit(x_train, y_train)
    pymilo_classification_test(one_class_svm, MODEL_NAME, (x_test, y_test))
