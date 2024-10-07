from beir.datasets.data_loader import GenericDataLoader
import json
from nltk.tokenize import sent_tokenize
import os
import sys
import time

from Ranking_TrialGPT import trialgpt_aggregation

if __name__ == "__main__":
  corpus = "Synthetic_Mass"
  model = "gpt-4-turbo"

	# the path of the matching results
  matching_results_path = "D:/Patient/matching_results_Synthetic_Mass_gpt-4-turbo.json"
  results = json.load(open(matching_results_path))

  # loading the trial2info dict
  trial2info = json.load(open("D:/Patient/trial_info.json"))

  # loading the patient info
  queries_path = "D:/Patient/queries.jsonl"
  queries = {}
  with open(queries_path, 'r') as f:
    for line in f:
      # Parse each line as JSON and append to the list
      queries_dict = (json.loads(line))
      queries[queries_dict["_id"]] = queries_dict["text"]

  # output file path
  output_path = f"D:/Patient/aggregation_results_{corpus}_{model}.json"

  if os.path.exists(output_path):
    output = json.load(open(output_path))
  else:
    output = {}

	# patient-level
  for patient_id, info in results.items():
		# get the patient note
    patient = queries[patient_id]
    sents = sent_tokenize(patient)
    sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
    sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
    patient = "\n".join(sents)

    if patient_id not in output:
      output[patient_id] = {}

		# label-level, 3 label / patient
    for label, trials in info.items():

			# trial-level
      for trial_id, trial_results in trials.items():
				# already cached results
        if trial_id in output[patient_id]:
          continue

        if type(trial_results) is not dict:
          output[patient_id][trial_id] = "matching result error"

          with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

          continue

				# specific trial information
        trial_info = trial2info[trial_id]

        try:
          result = trialgpt_aggregation(patient, trial_results, trial_info, model)
          output[patient_id][trial_id] = result

          with open(output_path, "w") as f:
            json.dump(output, f, indent=4)

        except:
          continue