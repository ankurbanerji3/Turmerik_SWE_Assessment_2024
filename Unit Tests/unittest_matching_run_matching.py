import unittest
import json
from Matching_run_matching import run_matching
from unittest.mock import patch

class TestMatchingRunMatching(unittest.TestCase):

    def setUp(self):
        # Load a sample of the JSON data (could be mocked)
        self.patient_trial_data = [
            {
                "patient_id": "sigir-20142",
                "patient": "An 8-year-old male presents with symptoms...",
                "trials": [
                    {
                        "brief_title": "Study on Bipolar Disorder",
                        "inclusion_criteria": "Clinical diagnosis of bipolar I disorder",
                        "exclusion_criteria": "History of neuroleptic malignant syndrome",
                        "drugs_list": ["Aripiprazole", "Lithium"],
                        "diseases_list": ["Bipolar Disorder Mania"]
                    }
                ]
            }
        ]

    @patch('Matching_run_matching.trialgpt_matching')
    def test_run_matching(self, mock_trialgpt_matching):
        # Mock the trial matching function
        mock_trialgpt_matching.return_value = "Patient is eligible for the trial."
        
        # Test the entire run_matching process
        result = run_matching(self.patient_trial_data)
        
        # Assert the structure of the output
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn('sigir-20142', result[0]['patient_id'])
        self.assertIn('eligible', result[0]['trials'][0]['matching_result'])

if __name__ == '__main__':
    unittest.main()