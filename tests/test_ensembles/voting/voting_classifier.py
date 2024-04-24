from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "VotingClassifier"

def voting_classifier():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    r1 = LogisticRegression(multi_class='multinomial', random_state=1)
    r2 = RandomForestClassifier(n_estimators=50, random_state=1)
    r3 = GaussianNB()
    voting_classifier = VotingClassifier([('lr', r1), ('rf', r2), ('r3', r3)], voting='hard').fit(x_train,y_train)
    pymilo_classification_test(voting_classifier, MODEL_NAME,(x_test, y_test))
