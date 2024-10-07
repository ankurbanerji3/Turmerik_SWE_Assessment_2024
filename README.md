Input files to start with -->
1. corpus.json

I would recommend to first try to run the notebook on platforms like Jupyter or Google Colab since it is quick, easy, and flexible to execute cell by cell.

NOTE: I chose to work with a very small "sample dataset" (not the actual large dataset) due to API token limits, and the patients in queries.json and trials in trial_info.json are taken by random. So it might be the case that some patients are ineligible for all the samples trials. Again, this is just for TESTING.

NOTE: Some unit test files may or may not pass. I've tested and successfully ran half of the unit tests but there are still some issues that I'm currenlty working on in attempts to pass all the unit tests.

----- Steps to run notebook -----
Head over to 

Prerequisites -->
1. Run Cell 2 then 3 to generate the list of queries.json.
2. Run Cell 11 to generate trials_info.json.

The first step is to run the "TrialGPT Retrieval" section -->
1. Run Cell 1, to install openai
2. In sequence, run Cell 4, 5, 6, 7, 8, 9, 10

The next step is to run the "TrialGPT Matching" section -->
1. In sequence, run Cell 12, 13, 14

The final step is to run the "TrialGPT Ranking" section -->
1. In sequence, run Cell 15, 16, 17, 18, 19, 20

The final eligibility results will be stored in the json file "eligibility_results.json" with the same format as given in the assessment doc.


----- Steps to run the directory files -----
Prerequisites -->
1. Run the files generate_queries.py and generate_trial_info.py to generate queries.json and trial_info.json files.
2. First step is to run the TrialGPT Retrieval phase to get a json containing the patients and the trials that are relevant according to keywords:
    - Run Retrieval_keyword_generation.py
    - Then run Retrieval_hybrid_fusion_retrieval.py
3. Second step is to run the TrialGPT Matching phase to judge which trials are actually a match for each patient.
    - Run Matching_TrialGPT.py
    - Run Matching_run_matching.py
4. Finally we have to run the TrialGPT Ranking phase to assign scores and rank each patient - trial pair from the previous phase
    - Run Ranking_TrialGPT.py
    - Run Ranking_run_aggregation.py
    - Run Ranking_run_results.py

Ranking_run_results.py will generate the eligibility results in the json file "eligibility_results.json".