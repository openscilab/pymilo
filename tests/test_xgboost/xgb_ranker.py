import numpy as np
import pytest

xgboost = pytest.importorskip("xgboost")
from xgboost import XGBRanker

from pymilo.utils.test_pymilo import pymilo_regression_test


RANKER_OBJECTIVES = [
    ("rank:pairwise", "XGBRanker-Pairwise"),
    ("rank:ndcg", "XGBRanker-NDCG"),
    ("rank:map", "XGBRanker-MAP"),
]


def _synthetic_ranking_dataset(n_queries=20, items_per_query=10, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_queries * items_per_query, n_features))
    y = (X[:, 0] + rng.normal(scale=0.1, size=X.shape[0])).round().clip(0, 4).astype(int)
    group = np.full(n_queries, items_per_query, dtype=int)
    return X, y, group


def xgb_ranker_all_objectives():
    X, y, group = _synthetic_ranking_dataset()
    n_train_queries = 16
    items = 10
    cut = n_train_queries * items
    x_train, y_train, group_train = X[:cut], y[:cut], group[:n_train_queries]
    x_test, y_test = X[cut:], y[cut:]

    # rank:map requires binary relevance labels (0/1 only).
    y_train_binary = (y_train > 0).astype(int)
    y_test_binary = (y_test > 0).astype(int)

    for objective, name in RANKER_OBJECTIVES:
        labels_train = y_train_binary if objective == "rank:map" else y_train
        labels_test = y_test_binary if objective == "rank:map" else y_test
        model = XGBRanker(
            objective=objective,
            n_estimators=8,
            max_depth=3,
            random_state=0,
        )
        model.fit(x_train, labels_train, group=group_train)
        assert pymilo_regression_test(model, name, (x_test, labels_test)) == True
