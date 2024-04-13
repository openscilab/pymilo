from sklearn.ensemble import HistGradientBoostingClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "HistGradientBoostingClassifier"

def hist_gradient_boosting_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    hist_gradient_boosting_classifier = HistGradientBoostingClassifier().fit(x_train, y_train)
    pymilo_classification_test(hist_gradient_boosting_classifier, MODEL_NAME, (x_test, y_test))
