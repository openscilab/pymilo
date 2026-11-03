import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBClassifier

from pymilo.utils.data_exporter import prepare_simple_classification_datasets
from pymilo.utils.test_pymilo import pymilo_classification_test, pymilo_regression_test


BINARY_OBJECTIVES = [
    ("binary:logistic", "XGBClassifier-BinaryLogistic"),
    ("binary:hinge", "XGBClassifier-BinaryHinge"),
]


def xgb_classifier_binary_all_objectives():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    for objective, name in BINARY_OBJECTIVES:
        model = XGBClassifier(
            objective=objective,
            n_estimators=8,
            max_depth=3,
            random_state=0,
        )
        model.fit(x_train, y_train)
        # binary:hinge yields discrete labels; hinge_loss metrics still apply via classification helper.
        # Prefer classification metrics for logistic; hinge can use either path.
        if objective == "binary:logistic":
            assert pymilo_classification_test(model, name, (x_test, y_test)) == True
        else:
            assert pymilo_regression_test(model, name, (x_test, y_test)) == True
