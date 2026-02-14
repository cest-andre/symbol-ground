import os
import re
import json
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModelForImageTextToText


hf_token = os.environ['HF_TOKEN']
carc_dir = '/home/alongon/data/ConceptARC'
num_tasks = 10

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-2-9b-it",
    token=hf_token
)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-9b-it",
    device_map="cuda:0",
    token=hf_token
)

# vlm_state_dict = torch.load('/home/alongon/model_weights/ewok/gemma_2_9b_llava_MORE.pth')
# rmv_ks = []
# for k in vlm_state_dict.keys():
#     if 'mm_projector' in k or 'vision_tower' in k:
#         rmv_ks.append(k)

# for k in rmv_ks:
#     del vlm_state_dict[k]

# # vlm_state_dict['lm_head.weight'] = vlm_state_dict['embed_tokens.weight']
# model.model.load_state_dict(vlm_state_dict)

for concept in os.listdir(carc_dir):
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
            input_str = task_string + f"input {test_num}: {format_str}\n\noutput {test_num}: "

            inputs = tokenizer(input_str, return_tensors="pt").to("cuda:0")
            outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.5, do_sample=True)

            #   TODO:  perhaps only specify the characters we want to keep when cleaning the string: [, ], space, and all numbers (though are all numbers used??)
            #          maybe split on the first character that isn't one of the accepted, and take first entry. 
            model_output = tokenizer.decode(outputs[0]).split(f'output {test_num}: ')[1].replace('\n', '').replace('<end_of_turn>', '').replace('<eos>', '')
            # model_output = re.sub('[^[] 0123456789]', '', model_output)
            gt_output = f"{task['test'][j]['output']}"[1:-1].replace(',', '')
            print(f'Model Output:\n{model_output}\n')
            print(f'Ground Truth:\n{gt_output}\n')
            print(f'Correct?  {model_output == gt_output}\n\n-----------\n\n')

    exit()