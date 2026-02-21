import os
import json
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText
from surprisal import CausalHuggingFaceModel


# hf_token = os.environ['HF_TOKEN']
carc_dir = '/home/alongon/data/ConceptARC'
num_tasks = 10
model_name = "google/gemma-2-2b-it"
results_dir = f'/home/alongon/data/carc_logprobs/{model_name.split("/")[1]}'
os.makedirs(results_dir, exist_ok=True)

# tokenizer = AutoTokenizer.from_pretrained(
#     model_name,
#     token=hf_token
# )
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="cuda:0",
#     token=hf_token
# )

model = CausalHuggingFaceModel(model_name, precision='bf16', trust_remote_code=True)

vlm_state_dict = torch.load('/home/alongon/model_weights/ewok/gemma_2_2b_llava_MORE.pth')
rmv_ks = []
for k in vlm_state_dict.keys():
    if 'mm_projector' in k or 'vision_tower' in k:
        rmv_ks.append(k)

for k in rmv_ks:
    del vlm_state_dict[k]

# vlm_state_dict['lm_head.weight'] = vlm_state_dict['embed_tokens.weight']
model.model.model.load_state_dict(vlm_state_dict)

results_dict = {}
for concept in os.listdir(carc_dir):
    # num_correct = 0
    # total_num = 0
    results_dict[concept] = []
    for i in range(num_tasks):
        with open(os.path.join(carc_dir, concept, f'{concept}{i+1}.json'), 'r') as file:
            task = json.load(file)

        task_string = "You are a helpful assistant that solves analogy making puzzles. Only give the answer, no other words or text.\n\nLet's try to complete the pattern:\n\n"
        for j in range(len(task['train'])):
            format_str = f"{task['train'][j]['input']}"[1:-1].replace(',', '')
            task_string += f"input {j+1}: {format_str}\n\n"
            
            format_str = f"{task['train'][j]['output']}"[1:-1].replace(',', '')
            task_string += f"output {j+1}: {format_str}\n\n"

        test_num = len(task['train']) + 1
        for j in range(len(task['test'])):
            format_str = f"{task['test'][j]['input']}"[1:-1].replace(',', '')
            format_str = task_string + f"input {test_num}: {format_str}\n\noutput {test_num}: "
            [result] = model.surprise(format_str, use_bos_token=False)
            idx = len(result.tokens)-1

            gt_output = f"{task['test'][j]['output']}"[1:-1].replace(',', '')
            input_str = format_str + gt_output
            [result] = model.surprise(input_str, use_bos_token=False)
            result = result.surprisals[idx:].sum()
            results_dict[concept].append(result)

            # #  NOTE:  response-based scoring using HF.
            # inputs = tokenizer(input_str, return_tensors="pt").to("cuda:0")
            # outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.5, do_sample=True)

            # model_output = tokenizer.decode(outputs[0]).split(f'output {test_num}: ')[1]#.split('input')[0][1:].split('Answer')[0].replace('\n', '').replace('<end_of_turn>', '').replace('<eos>', '').replace('<|eot_id|>', '')
            # for c in range(len(model_output)):
            #     if model_output[c] not in ['[', ']', ' ', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
            #         model_output = model_output[:c]
            #         break
            # model_output = model_output.strip()
            # print(f'Model Output:\n{model_output}\n')
            # print(f'Ground Truth:\n{gt_output}\n')
            # print(f'Correct?  {model_output == gt_output}\n\n-----------\n\n')

            # if model_output == gt_output:
            #     num_correct += 1

            # total_num += 1

    # print(f'{concept} Accuracy: {num_correct / total_num}')
    # print(f'{concept} LogProbs: {results_dict[concept]}')
with open(os.path.join(results_dir, 'vlm_logprobs.json'), 'w') as json_file:
    json.dump(results_dict, json_file)