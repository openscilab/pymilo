# -*- coding: utf-8 -*-
"""data exporter modules."""
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
