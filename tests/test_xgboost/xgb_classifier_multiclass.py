import pytest
from sklearn.datasets import load_iris

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBClassifier

from pymilo.utils.test_pymilo import pymilo_regression_test


MULTICLASS_OBJECTIVES = [
    ("multi:softmax", "XGBClassifier-MultiSoftmax"),
    ("multi:softprob", "XGBClassifier-MultiSoftprob"),
]


def _split_iris(threshold=20):
    X, y = load_iris(return_X_y=True)
    rng_idx = list(range(len(y)))
    interleaved = sorted(rng_idx, key=lambda i: (i % 3, i))
    X = X[interleaved]
    y = y[interleaved]
    return X[:-threshold], y[:-threshold], X[-threshold:], y[-threshold:]


def xgb_classifier_multiclass_all_objectives():
    x_train, y_train, x_test, y_test = _split_iris()
    for objective, name in MULTICLASS_OBJECTIVES:
        model = XGBClassifier(
            objective=objective,
            num_class=3,
            n_estimators=8,
            max_depth=3,
            random_state=0,
        )
        model.fit(x_train, y_train)
        # Use regression metrics on class labels (hinge_loss is binary-oriented).
        assert pymilo_regression_test(model, name, (x_test, y_test)) == True
