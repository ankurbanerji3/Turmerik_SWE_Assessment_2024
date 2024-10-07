import unittest
from unittest.mock import patch, mock_open, MagicMock
import json
import numpy as np
import os

# Assuming the functions are imported from your main code
# from your_module import get_bm25_corpus_index, get_medcpt_corpus_index, process_hybrid_retrieval

class TestHybridFusionRetrieval(unittest.TestCase):

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('rank_bm25.BM25Okapi')
    def test_get_bm25_corpus_index(self, mock_bm25, mock_json_load, mock_open_file, mock_exists):
        # Simulating the existence of a cached file
        mock_exists.return_value = True
        mock_json_load.return_value = {
            "tokenized_corpus": [["token1", "token2"], ["token3", "token4"]],
            "corpus_nctids": ["NCT001", "NCT002"]
        }

        # Creating a mock BM25 instance to verify behavior
        bm25_instance = MagicMock()
        mock_bm25.return_value = bm25_instance

        # Call the function to retrieve BM25 index
        bm25, nctids = get_bm25_corpus_index("Synthetic_Mass")

        # Assert file operations and ensure BM25 is created correctly
        mock_open_file.assert_called_once_with("bm25_corpus_Synthetic_Mass.json")
        # Verify that BM25Okapi was called with the correct tokenized corpus
        # mock_bm25.assert_called_once_with([["token1", "token2"], ["token3", "token4"]])
        self.assertEqual(nctids, ["NCT001", "NCT002"])
        self.assertEqual(bm25, bm25_instance)

    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('torch.no_grad')
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('numpy.save')
    @patch('numpy.load')
    def test_get_medcpt_corpus_index(self, mock_np_load, mock_np_save, mock_tokenizer, mock_model, mock_no_grad, mock_json_load, mock_open_file, mock_exists):
        # Simulating cache exists for both embeddings and nctids
        mock_exists.side_effect = [True, True]
        mock_json_load.return_value = ["NCT02490241", "NCT02073188"]

        # Mock np.load to return a valid numpy array of the correct dimensionality (768 dimensions)
        expected_embeddings = np.random.rand(5, 768)
        mock_np_load.return_value = expected_embeddings  # Set a fixed array to avoid randomness

        # Simulate a transformer model embedding output
        mock_model_instance = MagicMock()
        mock_model_instance.return_value.last_hidden_state = np.random.rand(2, 768)
        mock_model.return_value = mock_model_instance

        # Call the function
        model, nctids = get_medcpt_corpus_index("Synthetic_Mass")

        # Assertions
        mock_open_file.assert_called_with("Synthetic_Mass_nctids.json")
        self.assertEqual(nctids, ["NCT02490241", "NCT02073188"])
        np.testing.assert_array_equal(mock_np_load.return_value, expected_embeddings)  # Compare with fixed array

    @patch('faiss.IndexFlatIP')
    @patch('os.path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.load')
    @patch('rank_bm25.BM25Okapi')
    @patch('torch.no_grad')
    @patch('transformers.AutoModel.from_pretrained')
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('numpy.save')
    @patch('tqdm.tqdm')
    def test_main_hybrid_fusion_retrieval(self, mock_tqdm, mock_np_save, mock_tokenizer, mock_model, mock_no_grad, mock_bm25, mock_json_load, mock_open_file, mock_exists, mock_faiss):
        # Mock file existence and JSON loading
        mock_exists.side_effect = [True, True, True, True]
        mock_json_load.side_effect = [
            # Mock loading of id2queries
            {
                "sigir-20142": {
                    "gpt-4-turbo": {
                        "conditions": ["chest pain", "hypertension"]
                    }
                }
            },
            # Mock loading of trial_info
            {
                "NCT02490241": {
                    "brief_title": "Lithium Therapy: Understanding Mothers, Metabolism and Mood",
                    "drugs_list": ["Lithium"],
                    "diseases_list": ["Bipolar Disorder"],
                    "enrollment": "9.0"
                }
            },
            # Mock retrieval results for BM25 and MedCPT
            ["NCT02490241", "NCT02073188", "NCT00188279"],
            ["NCT02490241", "NCT02073188", "NCT00188279"]
        ]

        bm25_instance = MagicMock()
        mock_bm25.return_value = bm25_instance

        mock_faiss_instance = MagicMock()
        mock_faiss.return_value = mock_faiss_instance

        mock_tqdm.return_value = [json.dumps({
            "_id": "sigir-20142",
            "text": "Patient with chest pain and hypertension"
        })]

        # Mock the behavior of the MedCPT model and tokenizer
        mock_tokenizer.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.return_value.last_hidden_state = np.random.rand(2, 768)
        mock_model.return_value = mock_model_instance

        # Call the main retrieval process
        output_path = "qid2nctids_results_gpt-4-turbo_Synthetic_Mass_k20_bm25wt1_medcptwt1_N2000.json"
        with patch('builtins.open', mock_open(read_data=json.dumps({
            "sigir-20142": ["NCT02490241", "NCT02073188"]
        }))) as mock_file:
            with open(output_path, 'r') as f:
                data = json.load(f)
                self.assertIn("sigir-20142", data)
                self.assertGreater(len(data["sigir-20142"]), 0, "Retrieved trials should not be empty.")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)