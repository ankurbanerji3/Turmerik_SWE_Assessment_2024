from beir.datasets.data_loader import GenericDataLoader
import faiss
import json
from nltk import word_tokenize
import numpy as np
import os
from rank_bm25 import BM25Okapi
import sys
import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

"""
Conduct the first stage retrieval by the hybrid retriever
"""

def get_bm25_corpus_index(corpus):
  corpus_path = os.path.join(f"D:/Patient/bm25_corpus_{corpus}.json")

	# if already cached then load, otherwise build
  if os.path.exists(corpus_path):
    corpus_data = json.load(open(corpus_path))
    tokenized_corpus = corpus_data["tokenized_corpus"]
    corpus_nctids = corpus_data["corpus_nctids"]

  else:
    tokenized_corpus = []
    corpus_nctids = []

    with open(f"D:/Patient/corpus.jsonl", "r") as f:
      for line in f.readlines():
        entry = json.loads(line)
        corpus_nctids.append(entry["_id"])

        # weighting: 3 * title, 2 * condition, 1 * text
        tokens = word_tokenize(entry["title"].lower()) * 3
        for disease in entry["metadata"]["diseases_list"]:
          tokens += word_tokenize(disease.lower()) * 2
        tokens += word_tokenize(entry["text"].lower())

        tokenized_corpus.append(tokens)

    corpus_data = {
			"tokenized_corpus": tokenized_corpus,
			"corpus_nctids": corpus_nctids,
		}

    with open(corpus_path, "w") as f:
      json.dump(corpus_data, f, indent=4)

  bm25 = BM25Okapi(tokenized_corpus)

  return bm25, corpus_nctids


def get_medcpt_corpus_index(corpus):
  corpus_path = f"D:/Patient/{corpus}_embeds.npy"
  nctids_path = f"D:/Patient/{corpus}_nctids.json"

  if os.path.exists(corpus_path):
    embeds = np.load(corpus_path)
    corpus_nctids = json.load(open(nctids_path))

  else:
    embeds = []
    corpus_nctids = []

    model = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")

    with open(f"D:/Patient/corpus.jsonl", "r") as f:
      print("Encoding the corpus")
      for line in tqdm.tqdm(f.readlines()):
        entry = json.loads(line)
        corpus_nctids.append(entry["_id"])

        title = entry["title"]
        text = entry["text"]

        with torch.no_grad():
          # tokenize the articles
          encoded = tokenizer(
              [[title, text]],
              truncation=True,
              padding=True,
              return_tensors='pt',
              max_length=512,
          ).to("cuda")

          embed = model(**encoded).last_hidden_state[:, 0, :]

          embeds.append(embed[0].cpu().numpy())

    embeds = np.array(embeds)

    np.save(corpus_path, embeds)
    with open(nctids_path, "w") as f:
      json.dump(corpus_nctids, f, indent=4)

  index = faiss.IndexFlatIP(768)
  index.add(embeds)

  return index, corpus_nctids

if __name__ == '__main__':
  corpus = "Synthetic_Mass"
  q_type = "gpt-4-turbo"

  # different k for fusion
  k = 20

  # bm25 weight
  bm25_wt = 1

  # medcpt weight
  medcpt_wt = 1

  # how many to rank
  N = 2000

  id2queries = json.load(open(f"D:/Patient/id2queries.json"))

  trial_info = json.load(open(f"D:/Patient/trial_info.json"))

  # loading the indices
  bm25, bm25_nctids = get_bm25_corpus_index(corpus)
  medcpt, medcpt_nctids = get_medcpt_corpus_index(corpus)

  # loading the query encoder for MedCPT
  model = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").to("cuda")
  tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")

  # then conduct the searches, saving top 1k
  output_path = f"D:/Patient/qid2nctids_results_{q_type}_{corpus}_k{k}_bm25wt{bm25_wt}_medcptwt{medcpt_wt}_N{N}_2.json"

  qid2nctids = {}
  recalls = []

  retrieved_trials_final = []

  with open(f"D:/Patient/queries.jsonl", "r") as f:
    for line in tqdm.tqdm(f.readlines()):
      entry = json.loads(line)
      query = entry["text"]
      qid = entry["_id"]
      print(qid)

    if "turbo" in q_type:
      conditions = id2queries[qid][q_type]["conditions"]

    if len(conditions) == 0:
      nctid2score = {}

    else:
      # a list of nctid lists for the bm25 retriever
      bm25_condition_top_nctids = []

      for condition in conditions:
        tokens = word_tokenize(condition.lower())
        top_nctids = bm25.get_top_n(tokens, bm25_nctids, n=N)
        bm25_condition_top_nctids.append(top_nctids)

      # doing MedCPT retrieval
      with torch.no_grad():
        encoded = tokenizer(
            conditions,
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=256,
        ).to("cuda")

        # encode the queries (use the [CLS] last hidden states as the representations)
        embeds = model(**encoded).last_hidden_state[:, 0, :].cpu().numpy()

        # search the Faiss index
        scores, inds = medcpt.search(embeds, k=N)

      medcpt_condition_top_nctids = []
      for ind_list in inds:
        top_nctids = [medcpt_nctids[ind] for ind in ind_list]
        medcpt_condition_top_nctids.append(top_nctids)

      nctid2score = {}

      for condition_idx, (bm25_top_nctids, medcpt_top_nctids) in enumerate(zip(bm25_condition_top_nctids, medcpt_condition_top_nctids)):
        if bm25_wt > 0:
          for rank, nctid in enumerate(bm25_top_nctids):
            if nctid not in nctid2score:
              nctid2score[nctid] = 0

            nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))

        if medcpt_wt > 0:
          for rank, nctid in enumerate(medcpt_top_nctids):
            if nctid not in nctid2score:
              nctid2score[nctid] = 0

            nctid2score[nctid] += (1 / (rank + k)) * (1 / (condition_idx + 1))

    nctid2score = sorted(nctid2score.items(), key=lambda x: -x[1])
    top_nctids = [nctid for nctid, _ in nctid2score[:N]]
    qid2nctids[qid] = top_nctids

    print(qid2nctids[qid])

    retrieved_trials = {}
    retrieved_trials["patient_id"] = qid
    retrieved_trials["patient"] = query
    retrieved_trials["trials"] = []
    for trial in qid2nctids[qid]:
      retrieved_trials["trials"].append(trial_info[trial])

    retrieved_trials_final.append(retrieved_trials)

  with open(output_path, "w") as f:
    json.dump(qid2nctids, f, indent=4)

  with open("D:/Patient/retrieved_trials.json_2", "w") as f:
    json.dump(retrieved_trials_final, f, indent=4)