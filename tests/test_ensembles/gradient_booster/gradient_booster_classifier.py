from sklearn.ensemble import GradientBoostingClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "GradientBoostingClassifier"

def gradient_booster_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    gradient_booster_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train, y_train)
    pymilo_classification_test(gradient_booster_classifier, MODEL_NAME, (x_test, y_test))
