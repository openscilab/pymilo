from sklearn.neural_network import MLPClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Multi Layer Perceptron Classification"


def multi_layer_perceptron_classification():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    # Create MLPClassifier object
    multi_layer_perceptron_classifier = MLPClassifier(random_state=1, max_iter=500).fit(x_train, y_train)
    # Train the model using the training sets
    multi_layer_perceptron_classifier.fit(x_train, y_train)
    assert pymilo_classification_test(
        multi_layer_perceptron_classifier, MODEL_NAME, (x_test, y_test)) == True 
