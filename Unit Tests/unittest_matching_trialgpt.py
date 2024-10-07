import unittest
from Matching_TrialGPT import parse_criteria, get_matching_prompt, trialgpt_matching
from unittest.mock import patch


class TestMatchingTrialGPT(unittest.TestCase):

    def setUp(self):
        # Sample data to simulate real input for the methods
        self.trial_info = {
            'brief_title': 'Study on Bipolar Disorder',
            'inclusion_criteria': 'Clinical diagnosis of bipolar I disorder',
            'exclusion_criteria': 'History of neuroleptic malignant syndrome',
            'drugs_list': ['Aripiprazole', 'Lithium']
        }
        
        self.patient_data = {
            'patient_id': 'sigir-20142',
            'patient': '8-year-old male with symptoms of fever and dyspnea.',
        }
        
        self.inc_exc = {
            'inclusion': ['Clinical diagnosis of bipolar I disorder'],
            'exclusion': ['History of neuroleptic malignant syndrome']
        }

    def test_parse_criteria(self):
        # Test if the criteria are parsed correctly
        inclusion = 'Clinical diagnosis of bipolar I disorder'
        exclusion = 'History of neuroleptic malignant syndrome'
        
        parsed_criteria = parse_criteria({'inclusion_criteria': inclusion, 'exclusion_criteria': exclusion})
        self.assertIn('inclusion', parsed_criteria)
        self.assertIn('exclusion', parsed_criteria)
        self.assertEqual(parsed_criteria['inclusion'][0], inclusion)
        self.assertEqual(parsed_criteria['exclusion'][0], exclusion)

    def test_get_matching_prompt(self):
        # Test prompt generation for matching
        prompt = get_matching_prompt(self.trial_info, self.inc_exc, self.patient_data)
        self.assertIn('Study on Bipolar Disorder', prompt)
        self.assertIn('Clinical diagnosis of bipolar I disorder', prompt)
        self.assertIn('8-year-old male', prompt)
        self.assertIn('History of neuroleptic malignant syndrome', prompt)

    @patch('Matching_TrialGPT.call_gpt_model')
    def test_trialgpt_matching(self, mock_call_gpt_model):
        # Mock the GPT model response
        mock_call_gpt_model.return_value = "Patient is eligible for the trial."
        
        # Test the trial matching function
        result = trialgpt_matching(self.trial_info, self.patient_data, model='gpt-model')
        self.assertIn('eligible', result)

if __name__ == '__main__':
    unittest.main()