from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
from sklearn.neighbors import KNeighborsRegressor
from pymilo.utils.test_pymilo import pymilo_regression_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "VotingRegressor"

def voting_regressor():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    r1 = LinearRegression()
    r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    r3 = KNeighborsRegressor()
    voting_regressor = VotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)]).fit(x_train,y_train)
    pymilo_regression_test(voting_regressor, MODEL_NAME,(x_test, y_test))
