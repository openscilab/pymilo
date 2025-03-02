from export_exceptions import invalid_model
from export_exceptions import valid_model_invalid_structure_linear_model
from export_exceptions import valid_model_invalid_structure_neural_network
from export_exceptions import valid_model_irrelevant_chain

from import_exceptions import invalid_json, invalid_url, valid_url_invalid_file, valid_url_valid_file

EXCEPTION_TESTS = {
    'IMPORT': [
        invalid_json,
        invalid_url,
        valid_url_invalid_file,
        valid_url_valid_file,
        ],
    'EXPORT': [
        invalid_model,
        valid_model_invalid_structure_linear_model,
        valid_model_invalid_structure_neural_network,
        valid_model_irrelevant_chain
        ]
}

def test_full():
    for category in EXCEPTION_TESTS:
        category_all_test_pass = True
        for test in EXCEPTION_TESTS[category]:
            category_all_test_pass = category_all_test_pass and test()
            assert category_all_test_pass == True
            print("Test of Category: " + category + " with granularity of: " + test.__name__ + " executed successfully." )