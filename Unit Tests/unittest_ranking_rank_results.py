import unittest
from Ranking_rank_results import get_matching_score, get_agg_score, rank_trials
from unittest.mock import patch, mock_open
import json

class TestRankingRankResults(unittest.TestCase):

    def setUp(self):
        # Sample data simulating the content of the three input files

        # Sample matching_results_Synthetic_Mass_gpt-4-turbo.json content
        self.sample_matching_results = {
            "sigir-20142": {
                "trials": {
                    "NCT00665366": {
                        "inclusion": {
                            "0": ["No clinical diagnosis of bipolar I disorder.", [0], "not applicable"]
                        },
                        "exclusion": {
                            "0": ["Not applicable for 8-year-old male.", [], "not applicable"]
                        }
                    }
                }
            }
        }

        # Sample aggregation_results_Synthetic_Mass_gpt-4-turbo.json content
        self.sample_aggregation_results = {
            "sigir-20142": {
                "NCT00665366": {
                    "relevance_score_R": 10.0,
                    "eligibility_score_E": -10.0,
                    "eligibilityCriteriaMet": "No"
                }
            }
        }

        # Sample trial_info.json content
        self.sample_trial_info = {
            "NCT00665366": {
                "brief_title": "Study on Bipolar Disorder",
                "inclusion_criteria": "Clinical diagnosis of bipolar I disorder",
                "exclusion_criteria": "Women of childbearing potential..."
            }
        }

    def test_get_matching_score(self):
        # Test the get_matching_score function for computing the matching score
        matching = {
            "inclusion": {
                "0": ["No clinical diagnosis of bipolar I disorder.", [0], "not applicable"]
            },
            "exclusion": {
                "0": ["Not applicable for 8-year-old male.", [], "not applicable"]
            }
        }
        score = get_matching_score(matching)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)  # As there is no applicable match, score should be 0

    def test_get_agg_score(self):
        # Test the get_agg_score function for computing the aggregation score
        assessment = {
            "relevance_score_R": 10.0,
            "eligibility_score_E": -10.0,
            "eligibilityCriteriaMet": "No"
        }
        score = get_agg_score(assessment)
        self.assertIsInstance(score, float)
        self.assertEqual(score, 0.0)  # Since eligibility is negative, overall score should be 0

    @patch('Ranking_rank_results.load_json_file')
    def test_rank_trials(self, mock_load_json_file):
        # Mock the loading of the input JSON files
        mock_load_json_file.side_effect = [
            self.sample_matching_results,
            self.sample_aggregation_results,
            self.sample_trial_info
        ]

        # Run the ranking function
        result = rank_trials(
            matching_results_file='matching_results_Synthetic_Mass_gpt-4-turbo.json',
            aggregation_results_file='aggregation_results_Synthetic_Mass_gpt-4-turbo.json',
            trial_info_file='trial_info.json'
        )

        # Validate the output structure
        self.assertIsInstance(result, dict)
        self.assertIn('sigir-20142', result)
        self.assertIn('NCT00665366', result['sigir-20142']['trials'])
        trial_data = result['sigir-20142']['trials']['NCT00665366']

        # Check if the final trial result contains scores and a rank
        self.assertIn('final_score', trial_data)
        self.assertIn('rank', trial_data)
        self.assertEqual(trial_data['final_score'], 0.0)  # Based on our mock data
        self.assertEqual(trial_data['rank'], 1)  # Only one trial, so it should be rank 1

if __name__ == '__main__':
    unittest.main()
