# -*- coding: utf-8 -*-
"""data exporter modules."""
import numpy as np
from sklearn import datasets


def _split_X_y(X, y, threshold=20):
    """
    Split X and y into train and test sets.

    :param X: the data
    :type X: list or np.ndarray
    :param y: the targets
    :type y: list or np.ndarray
    :param threshold: threshold for train/test spliting
    :int threshold: int
    :return: X train, y train, X test, y test
    """
    X_train, X_test = X[:-threshold], X[-threshold:]
    y_train, y_test = y[:-threshold], y[-threshold:]
    return X_train, y_train, X_test, y_test


def prepare_simple_classification_datasets(threshold=50):
    """
    Generate a dataset for classification (breast cancer wisconsin).

    :param threshold: threshold for train/test spliting
    :int threshold: int
    :return: splited dataset for classification
    """
    cancer_X, cancer_y = datasets.load_breast_cancer(return_X_y=True)
    return _split_X_y(cancer_X, cancer_y, threshold)


def prepare_simple_regression_datasets(threshold=20):
    """
    Generate a dataset for regression (the diabetes).

    :param threshold: threshold for train/test spliting
    :int threshold: int
    :return: splited dataset for regression
    """
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    return _split_X_y(diabetes_X, diabetes_y, threshold)


def prepare_simple_ranking_datasets(n_queries=8, n_per_query=10, n_features=6, threshold_queries=2, random_state=42):
    """
    Generate a synthetic dataset for learning-to-rank tasks.

    :param n_queries: number of query groups
    :type n_queries: int
    :param n_per_query: number of documents per query
    :type n_per_query: int
    :param n_features: number of features
    :type n_features: int
    :param threshold_queries: number of trailing query groups used as the test split
    :type threshold_queries: int
    :param random_state: random seed
    :type random_state: int
    :return: X train, y train, qid train, X test, y test, qid test
    """
    rng = np.random.RandomState(random_state)
    n_samples = n_queries * n_per_query
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 5, size=n_samples)
    qid = np.repeat(np.arange(n_queries), n_per_query)
    split = n_samples - (threshold_queries * n_per_query)
    return X[:split], y[:split], qid[:split], X[split:], y[split:], qid[split:]


def prepare_simple_clustering_datasets():
    """
    Generate a dataset for clustering (the iris).

    :return: dataset for clustering
    """
    # Load the Iris dataset
    iris = datasets.load_iris()
    # Access the features and target
    X = iris.data  # Features
    y = iris.target  # Target (labels)
    return X, y
