####### add Pymilo to path.
from pathlib import Path
import os
import sys 
parent_path = Path(os.getcwd()).parent.absolute()
pymilo_path = os.path.join(parent_path,'pymilo')
sys.path.insert(0,pymilo_path)

# import Pymilo Export & Import
from pymilo.pymilo_obj import Export 
from pymilo.pymilo_obj import Import
# metrics for regression precision evaluation
from sklearn.metrics import mean_squared_error, r2_score
# metrics for classification precision evaluation
from sklearn.metrics import accuracy_score, hinge_loss
# compare the results before & after Pymilo
from pymilo.pymilo_func import compare_model_outputs

def test_pymilo(model, model_name, test_data):
    x_test,_ = test_data
    # Export model using Pymilo Exporter
    exported_model = Export(model)
    exported_model_serialized_path = os.path.join(os.getcwd(),"tests","exported_models",f'{model_name}.json')
    exported_model.save(exported_model_serialized_path)

    # Import pymilo-serialized model using Pymilo Importer
    imported_model = Import(exported_model_serialized_path)
    imported_sklearn_model = imported_model.to_model()
    return imported_sklearn_model.predict(x_test)

def test_pymilo_regression(regressor, model_name, test_data):
    x_test,y_test = test_data
    pre_pymilo_model_y_pred = regressor.predict(x_test)
    pre_pymilo_model_prediction_output = {
        "mean-error": mean_squared_error(y_test, pre_pymilo_model_y_pred),
        "r2-score": r2_score(y_test, pre_pymilo_model_y_pred)
    }
    post_pymilo_model_y_pred = test_pymilo(regressor,model_name,test_data)
    post_pymilo_model_prediction_outputs = {
        "mean-error": mean_squared_error(y_test, post_pymilo_model_y_pred),
        "r2-score": r2_score(y_test, post_pymilo_model_y_pred)
    }
    comparison_result = compare_model_outputs(pre_pymilo_model_prediction_output,post_pymilo_model_prediction_outputs)
    if(comparison_result):
        print(f'Pymilo Test for Model:{model_name} succeed✅.')
    else:
        print(f'Pymilo Test for Model:{model_name} failed❌.')
    return comparison_result

def test_pymilo_classification(classifier, model_name, test_data):
    x_test,y_test = test_data
    pre_pymilo_model_y_pred = classifier.predict(x_test)
    pre_pymilo_model_prediction_output = {
        "accuracy-score": accuracy_score(y_test, pre_pymilo_model_y_pred),
        "hinge-loss": hinge_loss(y_test,pre_pymilo_model_y_pred)
    }
    post_pymilo_model_y_pred = test_pymilo(classifier,model_name,test_data)
    post_pymilo_model_prediction_outputs =  {
        "accuracy-score": accuracy_score(y_test, post_pymilo_model_y_pred),
        "hinge-loss": hinge_loss(y_test,post_pymilo_model_y_pred)
    }
    return compare_model_outputs(pre_pymilo_model_prediction_output,post_pymilo_model_prediction_outputs)
