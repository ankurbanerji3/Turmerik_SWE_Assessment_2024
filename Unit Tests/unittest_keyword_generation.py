import unittest
from unittest.mock import patch, MagicMock
import json
import os

# Assuming the function to be tested is defined in a module called 'your_module'
from Retrieval_keyword_generation import process_patient_queries

class TestProcessPatientQueries(unittest.TestCase):
    
    @patch('openai.ChatCompletion.create')
    def test_process_patient_queries(self, mock_openai):
        # Mock the response from the OpenAI API
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content=json.dumps({
                "summary": "Patient has chest pain, nausea, and dyspnea.",
                "conditions": ["Chest pain", "Hypertension", "Obesity"]
            })))
        ]
        mock_openai.return_value = mock_response

        # Prepare a sample input file path and test data
        input_file = 'D:/Patient/test_queries.jsonl'
        test_data = [
            {
                "_id": "sigir-20141",
                "text": "A 58-year-old woman with chest pain, nausea, and dyspnea."
            }
        ]

        # Create the test input file
        with open(input_file, 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        # Call the function under test
        process_patient_queries(input_file, model='gpt-4o')

        # Check if the expected output files were created and contain the correct data
        with open(f"D:/Patient/retrieval_keywords_gpt-4o_2.json", 'r') as f:
            retrieval_output = json.load(f)
        
        with open(f"D:/Patient/id2queries_2.json", 'r') as f:
            id2queries_output = json.load(f)
        
        for patient_id, data in retrieval_output.items():
            self.assertIn("summary", data)
            self.assertTrue(data["summary"], "Summary should not be empty")
            self.assertIn("conditions", data)
            self.assertTrue(data["conditions"], "Conditions should not be empty")
            self.assertTrue(all(condition for condition in data["conditions"]), "Each condition should not be empty")
        
        # Verify that none of the fields are empty in id2queries_2.json
        for patient_id, data in id2queries_output.items():
            self.assertIn("raw", data)
            self.assertTrue(data["raw"], "Raw text should not be empty")
            self.assertIn("gpt-4-turbo", data)
            self.assertIn("summary", data["gpt-4-turbo"])
            self.assertTrue(data["gpt-4-turbo"]["summary"], "Summary in gpt-4-turbo should not be empty")
            self.assertIn("conditions", data["gpt-4-turbo"])
            self.assertTrue(data["gpt-4-turbo"]["conditions"], "Conditions in gpt-4-turbo should not be empty")
            self.assertTrue(all(condition for condition in data["gpt-4-turbo"]["conditions"]), "Each condition in gpt-4-turbo should not be empty")
        
        # Cleanup the test files
        os.remove(input_file)
        os.remove(f"D:/Patient/retrieval_keywords_gpt-4o_2.json")
        os.remove(f"D:/Patient/id2queries_2.json")

if __name__ == '__main__':
    unittest.main()
