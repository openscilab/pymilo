from sklearn.ensemble import AdaBoostClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "AdaBoostClassifier"

def adaboost_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    adaboost_classifier = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0).fit(x_train, y_train)
    pymilo_classification_test(adaboost_classifier, MODEL_NAME, (x_test, y_test))
