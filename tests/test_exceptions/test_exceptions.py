from export_exceptions import invalid_model, valid_model_invalid_structure_linear_model, valid_model_invalid_structure_neural_network, valid_model_irrelevant_chain
from import_exceptions import invalid_json

EXCEPTION_TESTS = {
    'IMPORT': [invalid_json],
    'EXPORT': [
        invalid_model,
        valid_model_invalid_structure_linear_model,
        valid_model_invalid_structure_neural_network, 
        valid_model_irrelevant_chain
        ]
}

def test_full():
    for category in EXCEPTION_TESTS.keys():
        category_all_test_pass = True
        for test in EXCEPTION_TESTS[category]:
            category_all_test_pass = category_all_test_pass and test()
            assert category_all_test_pass == True 
            print("Test of Category: " + category + " with granularity of: " + test.__name__ + " executed successfully." )