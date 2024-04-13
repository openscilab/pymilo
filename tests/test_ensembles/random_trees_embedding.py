from sklearn.ensemble import RandomTreesEmbedding
from pymilo.utils.test_pymilo import pymilo_regression_test, pymilo_test
from pymilo.utils.data_exporter import prepare_simple_regression_datasets

MODEL_NAME = "RandomTreesEmbedding"

def random_trees_embedding():
    x_train, y_train, x_test, y_test = prepare_simple_regression_datasets()
    random_trees_embedding = RandomTreesEmbedding(n_estimators=5, random_state=0, max_depth=1).fit(x_train, y_train)
    pymilo_test(random_trees_embedding, MODEL_NAME)

