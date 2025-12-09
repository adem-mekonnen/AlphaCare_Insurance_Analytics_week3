import unittest

class TestSanity(unittest.TestCase):
    def test_environment_works(self):
        """
        Simple test to ensure GitHub Actions can run Python
        and libraries are installed.
        """
        try:
            import pandas
            import numpy
            import xgboost
        except ImportError as e:
            self.fail(f"Dependency missing: {e}")

        self.assertTrue(True)

    def test_math(self):
        """
        A basic dummy test to get a 'Pass' status.
        """
        self.assertEqual(1 + 1, 2)

if __name__ == '__main__':
    unittest.main()