# -*- coding: utf-8 -*-
"""Helpers for XGBoost tests. All training and prediction stay on CPU."""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

xgboost_available = False
try:
    import xgboost
    from pymilo.pymilo_param import XGBOOST_MODEL_TABLE, NOT_SUPPORTED
    xgboost_available = XGBOOST_MODEL_TABLE.get("XGBClassifier") != NOT_SUPPORTED
except Exception:
    xgboost = None
    xgboost_available = False


def cpu_estimator_kwargs(estimator_cls, **overrides):
    """
    Build constructor kwargs that force CPU execution.

    :param estimator_cls: XGBoost sklearn estimator class
    :type estimator_cls: type
    :param overrides: extra constructor kwargs
    :return: kwargs dict
    """
    params = {
        "n_estimators": 8,
        "max_depth": 2,
        "random_state": 0,
        "verbosity": 0,
        "n_jobs": 1,
        "device": "cpu",
        "tree_method": "hist",
    }
    params.update(overrides)
    return params
