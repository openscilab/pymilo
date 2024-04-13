from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "StackingClassifier"

def stacking_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
        ('svr', make_pipeline(LinearSVC(dual="auto", random_state=42)))
        ]
    stacking_classifier = StackingClassifier(
        estimators=estimators, final_estimator=LogisticRegression()
    ).fit(x_train, y_train)

    pymilo_classification_test(stacking_classifier, MODEL_NAME, (x_test, y_test))

