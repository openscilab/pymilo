from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "StackingRegressor"

def stacking_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    estimators = [
        ('lr', RidgeCV()),
        ('svr', LinearSVR(random_state=42))]
    stacking_regressor = StackingRegressor(
        estimators=estimators,
        final_estimator=RandomForestRegressor(n_estimators=10,random_state=42)).fit(x_train,y_train)
    pymilo_regression_test(stacking_regressor, MODEL_NAME,(x_test, y_test))
