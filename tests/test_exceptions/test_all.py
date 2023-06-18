import os
import unittest

from test_export_exceptions import invalid_model, valid_model_invalid_structure
from test_import_exceptions import invalid_json

class TestExceptionalCases(unittest.TestCase):

    EXCEPTION_TESTS = {
        'IMPORT': [invalid_json],
        'EXPORT': [invalid_model, valid_model_invalid_structure]
    }

    def test_full(self):
        for category in self.EXCEPTION_TESTS.keys():
            category_all_test_pass = True
            for test in self.EXCEPTION_TESTS[category]:
                category_all_test_pass = category_all_test_pass and test()
                self.assertTrue(category_all_test_pass)
                print("Test of Category: " + category + " with granularity of: " + test.__name__ + " executed successfully." )

if __name__ == '__main__':
    unittest.main()
