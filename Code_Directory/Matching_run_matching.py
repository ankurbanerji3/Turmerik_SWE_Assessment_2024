"""
Running the TrialGPT matching for three cohorts (sigir, TREC 2021, TREC 2022).
"""

import json
from nltk.tokenize import sent_tokenize
import os
import sys

from Matching_TrialGPT import trialgpt_matching

corpus = "Synthetic_Mass"
model = "gpt-4-turbo"

dataset = json.load(open(f"D:/Patient/retrieved_trials.json"))

output_path = f"D:/Patient/matching_results_{corpus}_{model}.json"

# Dict{Str(patient_id): Dict{Str(label): Dict{Str(trial_id): Str(output)}}}
if os.path.exists(output_path):
	output = json.load(open(output_path))
else:
	output = {}

for instance in dataset:
  # Dict{'patient': Str(patient), '0': Str(NCTID), ...}
  patient_id = instance["patient_id"]
  patient = instance["patient"]
  sents = sent_tokenize(patient)
  sents.append("The patient will provide informed consent, and will comply with the trial protocol without any practical issues.")
  sents = [f"{idx}. {sent}" for idx, sent in enumerate(sents)]
  patient = "\n".join(sents)

  # initialize the patient id in the output
  if patient_id not in output:
    output[patient_id] = {"trials": {}}

	# for label in ["2", "1", "0"]:
	# 	if label not in instance: continue

  for trial in instance["trials"]:
    trial_id = trial["NCTID"]

    # already calculated and cached
    if trial_id in output[patient_id]["trials"]:
      continue

    # in case anything goes wrong (e.g., API calling errors)
    try:
      results = trialgpt_matching(trial, patient, model)
      output[patient_id]["trials"][trial_id] = results

      with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    except Exception as e:
      print(e)
      continue