import os
import json
import matplotlib.pyplot as plt


model_name = "google/gemma-2-2b-it"
results_dir = f'/home/alongon/data/carc_logprobs/{model_name.split("/")[1]}'
save_dir = '/home/alongon/figures/concept_arc'

with open(os.path.join(results_dir, 'llm_logprobs.json')) as json_file:
    llm_results = json.load(json_file)

with open(os.path.join(results_dir, 'vlm_logprobs.json')) as json_file:
    vlm_results = json.load(json_file)

for concept in llm_results.keys():
    vlm_count = 0
    for i in range(len(llm_results[concept])):
        if vlm_results[concept][i] > llm_results[concept][i]:
            vlm_count += 1

    print(f'VLM Higher % for Concept {concept}:  {vlm_count / len(llm_results[concept])}')