from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.util import has_named_parameter

MODEL_NAME = "BaggingClassifier"

def bagging_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    if has_named_parameter(BaggingClassifier, "estimator"):
        bagging_classifier = BaggingClassifier(estimator=SVC(), n_estimators=10, random_state=0).fit(x_train, y_train)
    else:
        bagging_classifier = BaggingClassifier(n_estimators=10, random_state=0).fit(x_train, y_train)
    pymilo_classification_test(bagging_classifier, MODEL_NAME, (x_test, y_test))
