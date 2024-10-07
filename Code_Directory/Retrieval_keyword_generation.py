"""
generate the search keywords for each patient
"""

from openai import OpenAI
import json

client = OpenAI(
    # This is the default and can be omitted
		# api_key = "Your API Key"
		api_key = 'Your API Key'
)


def get_keyword_generation_messages(note):
	system = 'You are a helpful assistant and your task is to help search relevant clinical trials for a given patient description. Please first summarize the main medical problems of the patient. Then generate up to 32 key conditions for searching relevant clinical trials for this patient. The key condition list should be ranked by priority. Please output only a JSON dict formatted as Dict{{"summary": Str(summary), "conditions": List[Str(condition)]}}.'

	prompt =  f"Here is the patient description: \n{note}\n\nJSON output:"

	messages = [
		{"role": "system", "content": system},
		{"role": "user", "content": prompt}
	]

	return messages

# def process_patient_queries(input_file, model='gpt-4o'):
#    outputs = {}
#    ret_trials = {}
#    # f"D:/Patient/queries.jsonl"
#    with open(input_file, "r") as f:
#       for line in f.readlines():
#        entry = json.loads(line)
#        messages = get_keyword_generation_messages(entry["text"])
       
#        response = client.chat.completions.create(
#           model='gpt-4o',
#           messages=messages,
#           temperature=0,
#         )
       
#        output = response.choices[0].message.content
#        output = output.strip("`").strip("json")
       
#        ret_trials[entry["_id"]] = {}
#        ret_trials[entry["_id"]]["raw"] = entry["text"]
#        ret_trials[entry["_id"]]["gpt-4-turbo"] = json.loads(output)
       
#        outputs[entry["_id"]] = json.loads(output)
       
#        with open(f"D:/Patient/retrieval_keywords_{model}_2.json", "w") as f:
#         json.dump(outputs, f, indent=4)
       
#        with open(f"D:/Patient/id2queries_2.json", "w") as f:
#         json.dump(ret_trials, f, indent=4)


if __name__ == '__main__':
   outputs = {}
   ret_trials = {}
   model = 'gpt-4o'
   
   with open(f"D:/Patient/queries.jsonl", "r") as f:
      for line in f.readlines():
       entry = json.loads(line)
       messages = get_keyword_generation_messages(entry["text"])
       
       response = client.chat.completions.create(
          model='gpt-4o',
          messages=messages,
          temperature=0,
        )
       
       output = response.choices[0].message.content
       output = output.strip("`").strip("json")
       
       ret_trials[entry["_id"]] = {}
       ret_trials[entry["_id"]]["raw"] = entry["text"]
       ret_trials[entry["_id"]]["gpt-4-turbo"] = json.loads(output)
       
       outputs[entry["_id"]] = json.loads(output)
       
       with open(f"D:/Patient/retrieval_keywords_{model}_2.json", "w") as f:
        json.dump(outputs, f, indent=4)
       
       with open(f"D:/Patient/id2queries_2.json", "w") as f:
        json.dump(ret_trials, f, indent=4)
   
   # process_patient_queries("D:/Patient/queries.jsonl")