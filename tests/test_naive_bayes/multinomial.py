from sklearn.naive_bayes import MultinomialNB

from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "MultinomialNB"

def multinomial_naive_bayes():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    multinomial_naive_bayes = MultinomialNB().fit(x_train, y_train)
    pymilo_classification_test(multinomial_naive_bayes, MODEL_NAME, (x_test, y_test))
