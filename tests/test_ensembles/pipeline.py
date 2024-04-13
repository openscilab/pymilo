from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from pymilo.utils.test_pymilo import pymilo_classification_test
from pymilo.utils.data_exporter import prepare_simple_classification_datasets

MODEL_NAME = "Pipeline"

def pipeline():
    x_train, y_train, x_test, y_test = prepare_simple_classification_datasets()
    pipeline = Pipeline([
        #('scaler', StandardScaler()), 
        ('svc', SVC())]).fit(x_train, y_train)
    pymilo_classification_test(pipeline, MODEL_NAME, (x_test, y_test))
