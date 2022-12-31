from sklearn import datasets


def split_x_y(x, y, threshold=20):
    # Split the data into training/testing sets
    x_train, x_test = x[:-1 * threshold], x[-1 * threshold:]
    # Split the targets into training/testing sets
    y_train, y_test = y[:-1 * threshold], y[-1 * threshold:]
    return x_train, y_train, x_test, y_test


def prepare_simple_classification_datasets(threshold=50):
    cancer_X, cancer_y = datasets.load_breast_cancer(return_X_y=True)
    return split_x_y(cancer_X, cancer_y, threshold)


def prepare_simple_regression_datasets(threshold=20):
    diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
    return split_x_y(diabetes_X, diabetes_y, threshold)


def prepare_logistic_regression_datasets(threshold=None):
    iris_X, iris_y = datasets.load_iris(return_X_y=True)
    threshold = threshold if (threshold) else len(iris_y) // 2
    return split_x_y(iris_X, iris_y, threshold)
