import json

# Load the JSONL file
input_file_path = 'D:/Patient/corpus.jsonl'
output_file_path = 'D:/Patient/trial_info.json'

# Prepare a dictionary for the output data
trial_info = {}

# Open and process the input file
with open(input_file_path, 'r') as f:
    for line in f:
        # Load each line as a dictionary
        entry = json.loads(line)

        # Extract the _id and metadata
        trial_id = entry["_id"]
        metadata = entry["metadata"]

        # Add the NCTID field in the metadata
        metadata["NCTID"] = trial_id

        # Add this information to the trial_info dictionary
        trial_info[trial_id] = metadata

# Save the trial_info dictionary to a new JSON file
with open(output_file_path, 'w') as out_file:
    json.dump(trial_info, out_file, indent=4)

print(f"trial_info.json has been created at: {output_file_path}")