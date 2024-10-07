import unittest
import json
from Ranking_run_aggregation import run_aggregation
from unittest.mock import patch, mock_open

class TestRankingRunAggregation(unittest.TestCase):

    def setUp(self):
        # Sample data simulating the content of the three input files
        
        # Sample queries.jsonl content (simulating line-by-line JSON objects)
        self.sample_queries_jsonl = [
            '{"patient_id": "sigir-20142", "note": "Patient presents with fever and dyspnea."}\n'
        ]
        
        # Sample trial_info.json content
        self.sample_trial_info = {
            "NCT00665366": {
                "brief_title": "Study on Bipolar Disorder",
                "inclusion_criteria": "Clinical diagnosis of bipolar I disorder",
                "exclusion_criteria": "Women of childbearing potential...",
                "drugs_list": ["Aripiprazole", "Lithium"],
                "diseases_list": ["Bipolar Disorder Mania"]
            }
        }
        
        # Sample matching_results_Synthetic_Mass_gpt-4-turbo.json content
        self.sample_matching_results = {
            "sigir-20142": {
                "trials": {
                    "NCT00665366": {
                        "inclusion": {
                            "0": ["No clinical diagnosis of bipolar I disorder.", [0], "not included"]
                        },
                        "exclusion": {
                            "0": ["Not applicable for 8-year-old male.", [], "not applicable"]
                        }
                    }
                }
            }
        }
    
    @patch('builtins.open', new_callable=mock_open, read_data=''.join(self.sample_queries_jsonl))
    def test_load_queries_jsonl(self, mock_file):
        # Test reading queries.jsonl
        with open('queries.jsonl') as f:
            lines = f.readlines()
        
        # Check if file reads correctly
        self.assertEqual(len(lines), 1)
        self.assertIn('sigir-20142', lines[0])
    
    @patch('Ranking_run_aggregation.load_json_file')
    def test_load_trial_info(self, mock_load_json_file):
        # Mocking the loading of trial_info.json
        mock_load_json_file.return_value = self.sample_trial_info
        
        trial_info = mock_load_json_file('trial_info.json')
        self.assertIn('NCT00665366', trial_info)
        self.assertEqual(trial_info['NCT00665366']['brief_title'], "Study on Bipolar Disorder")
    
    @patch('Ranking_run_aggregation.load_json_file')
    def test_load_matching_results(self, mock_load_json_file):
        # Mocking the loading of matching_results_Synthetic_Mass_gpt-4-turbo.json
        mock_load_json_file.return_value = self.sample_matching_results
        
        matching_results = mock_load_json_file('matching_results_Synthetic_Mass_gpt-4-turbo.json')
        self.assertIn('sigir-20142', matching_results)
        self.assertIn('NCT00665366', matching_results['sigir-20142']['trials'])
    
    @patch('Ranking_TrialGPT.trialgpt_aggregation')
    def test_aggregation_process(self, mock_trialgpt_aggregation):
        # Mock the GPT aggregation process
        mock_trialgpt_aggregation.return_value = {
            "relevance_score": 0.85,
            "eligibility_score": 0.75
        }
        
        # Simulate aggregation logic
        result = run_aggregation(
            queries_file='queries.jsonl',
            trial_info_file='trial_info.json',
            matching_results_file='matching_results_Synthetic_Mass_gpt-4-turbo.json'
        )
        
        # Check if aggregation was run for the patient and trial
        self.assertEqual(len(result), 1)
        self.assertIn('sigir-20142', result)
        self.assertIn('NCT00665366', result['sigir-20142']['trials'])
        self.assertIn('relevance_score', result['sigir-20142']['trials']['NCT00665366'])
        self.assertEqual(result['sigir-20142']['trials']['NCT00665366']['relevance_score'], 0.85)
        self.assertEqual(result['sigir-20142']['trials']['NCT00665366']['eligibility_score'], 0.75)

if __name__ == '__main__':
    unittest.main()
