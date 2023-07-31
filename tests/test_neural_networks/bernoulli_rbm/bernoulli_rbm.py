import os

from sklearn import metrics
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression

from pymilo.pymilo_obj import Export
from pymilo.pymilo_obj import Import
from pymilo.utils.test_pymilo import pymilo_export_path
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Bernoulli Restricted Boltzmann Machine (RBM)"

def bernoulli_rbm():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create Bernoulli RBM object
    logistic = LogisticRegression(solver="newton-cg", tol=1)
    rbm = BernoulliRBM(random_state=0, verbose=True)
    rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

    # Hyper-parameters. These were set by cross-validation,
    # using a GridSearchCV. Here we are not performing cross-validation to
    # save time.
    rbm.learning_rate = 0.06
    rbm.n_iter = 10

    # More components tend to give better prediction performance, but larger
    # fitting time
    rbm.n_components = 100
    logistic.C = 6000

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(x_train, y_train)

    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.0
    raw_pixel_classifier.fit(x_train, y_train)

    Y_pred = rbm_features_classifier.predict(x_test)
    before_report = metrics.classification_report(y_test, Y_pred)

    export_model_path = pymilo_export_path(rbm)
    exported_model = Export(rbm)
    exported_model_serialized_path = os.path.join(
        os.getcwd(), "tests", export_model_path, MODEL_NAME + '.json')
    exported_model.save(exported_model_serialized_path)

    imported_model = Import(exported_model_serialized_path)
    imported_rbm = imported_model.to_model()

    logistic = LogisticRegression(solver="newton-cg", tol=1)
    rbm_features_classifier = Pipeline(steps=[("rbm", imported_rbm), ("logistic", logistic)])
    logistic.C = 6000

    # Training RBM-Logistic Pipeline
    rbm_features_classifier.fit(x_train, y_train)

    # Training the Logistic regression classifier directly on the pixel
    raw_pixel_classifier = clone(logistic)
    raw_pixel_classifier.C = 100.0
    raw_pixel_classifier.fit(x_train, y_train)

    Y_pred = rbm_features_classifier.predict(x_test)
    after_report = metrics.classification_report(y_test, Y_pred)

    assert before_report == after_report





