import os
import json
from copy import deepcopy as copy
from typing import List

from tqdm import tqdm
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from template import TEMPLATES
# from llama import Dialog, Llama


def write_json(obj, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for line in obj:
            f.write(json.dumps(line) + '\n')


def read_jsonl(filename: str):
    output = []
    with open(filename, 'r', encoding='utf-8') as f:
        for idx, line in tqdm(enumerate(f)):
            line = line.strip()
            line = json.loads(line)
            output.append(line)
    return output


def load_jsonl(file_path):
    "General function to load jsonl file"
    _data = []
    with open(file_path, 'r') as f:
        for data in f:
            jline = json.loads(data)
            _data.append(jline)
    return _data


def apply_template(prior, template_type=0, option_type='easy', shot=0):
    def make_shot(indices):
        if shot == 0:
            return ''
        idx_all = np.random.choice([i for i in list(range(len(prior))) if i != indices], shot)
        instance_statements = ''
        for idx in idx_all:
            tmp_shot = copy(prior[idx])
            tmp_label = tmp_shot[f"options_{option_type}"][0]
            tmp_options = tmp_shot[f"options_{option_type}"][1:]
            np.random.shuffle(tmp_options)
            label = np.random.randint(4)
            tmp_options.insert(label, tmp_label)
            tmp_options = '- ' + '\n- '.join(tmp_options)
            instance_statement = f"\nGiven\n[RESPONSE]: {tmp_shot['condition']}\n[INSTRUCTION]:\n{tmp_options}\nYou must choose: {tmp_label}\n"
            instance_statements = instance_statements + instance_statement
        return f'\nFor instance: \n{instance_statements}\n'

    def apply_template_single(processed_item, now_idx, template_type):
        if isinstance(template_type, int):
            template = TEMPLATES[template_type]
        elif template_type == 'random':
            template = TEMPLATES[np.random.choice(list(TEMPLATES.keys()))]
        elif template_type == 'long':
            long_template = [0, 1, 2, 3, 4, 5, 6, 7]
            template = TEMPLATES[np.random.choice(long_template)]
        elif template_type == 'short':
            short_template = [8, 9, 10, 11, 12, 13, 14, 15]
            template = TEMPLATES[np.random.choice(short_template)]
        elif template_type == 'firstop':
            fisrtop_template = [4, 5, 6, 7, 10, 11, 14, 15]
            template = TEMPLATES[np.random.choice(fisrtop_template)]
        elif template_type == 'firstcd':
            fisrtcd_template = [0, 1, 2, 3, 8, 9, 12, 13]
            template = TEMPLATES[np.random.choice(fisrtcd_template)]
        else:
            raise Exception('Unknown template type')

        tmp = {
            'condition': processed_item['condition'],
            'option_mark': '-',
            'cot': '',
            'shot': make_shot(now_idx)
        }

        label_instruction = copy(processed_item[f"options_{option_type}"][0])
        options = copy(processed_item[f"options_{option_type}"][1:])
        np.random.shuffle(options)
        label = np.random.randint(4)
        options.insert(label, label_instruction)

        for i in range(1, 5):
            tmp[f"option{i}"] = options[i-1]
        processed_item['templated'] = template.format(**tmp)
        processed_item['template_type'] = template_type
        processed_item['options'] = options
        processed_item['label'] = label
        return processed_item

    processed = [apply_template_single(item, idx, template_type) for idx, item in enumerate(prior)]
    messages = [[{"role": "user", "content": item['templated']}] for item in processed]
    return processed, messages


def hf_call(model_arg):
    hf_arg = {
        'mistral': "mistralai/Mistral-7B-Instruct-v0.2",
        'solar': "upstage/SOLAR-10.7B-Instruct-v1.0",
        'llama2': "meta-llama/Llama-2-7b-chat-hf",
        'llama2_13': "meta-llama/Llama-2-13b-chat-hf",
        'llama2_70': "meta-llama/Llama-2-70b-chat-hf",
        'gemma': "google/gemma-7b-it",
        'commandr': 'CohereForAI/c4ai-command-r-v01',
        'mixtral': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
        'phi3': 'microsoft/Phi-3-mini-128k-instruct',
        'llama3': 'meta-llama/Meta-Llama-3-8B-Instruct'
    }[model_arg]

    if model_arg == 'solar':
        model = AutoModelForCausalLM.from_pretrained(hf_arg, device_map="auto", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(hf_arg)
    elif model_arg == 'phi3':
        model = AutoModelForCausalLM.from_pretrained(
            hf_arg,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_arg)
        model = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(hf_arg, device_map="auto")
        tokenizer = AutoTokenizer.from_pretrained(hf_arg)

    return model, tokenizer


if __name__ == '__main__':
    print(':)')
    print(';)')
