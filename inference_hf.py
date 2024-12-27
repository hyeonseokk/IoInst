import json
import os
import time
# import tiktoken
import argparse

from setproctitle import setproctitle
from os.path import join
from openai import OpenAI
from tqdm import tqdm
import torch

from utils import load_jsonl, read_jsonl, apply_template, hf_call

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # the device to load the model onto


class HF:
    def __init__(self, model, tokenizer, model_arg):
        self.model = model
        self.tokenizer = tokenizer
        self.model_arg = model_arg

    def generate_by_arg(self, encoded, generation_args):
        # print(f"encoded: {encoded}")
        if generation_args == 'greedy':
            if self.model_arg in ['commandr', 'solar']:
                return self.model.generate(
                    **encoded,
                    max_new_tokens=1024,
                    do_sample=False,
                    early_stopping=True,
                    use_cache=True
                )
            elif self.model_arg in ['phi3']:
                temporal_args = {
                    "max_new_tokens": 1024,
                    "return_full_text": False,
                    "temperature": 0.0,
                    "do_sample": False,
                }
                return self.model(encoded, **temporal_args)[0]['generated_text']
            elif self.model_arg in ['mixtral']:
                return self.model.generate(
                    encoded,
                    max_new_tokens=1024,
                    do_sample=False,
                )
            return self.model.generate(
                encoded,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True
            )

        elif generation_args.startswith('temp'):
            temperature = float(generation_args.split('temp')[-1])
            if self.model_arg in ['solar', 'commandr']:
                return self.model.generate(
                    **encoded,
                    max_new_tokens=1024,
                    do_sample=True,
                    temperature=temperature,
                    use_cache=True
                )
            elif self.model_arg in ['phi3']:
                temporal_args = {
                    "max_new_tokens": 1024,
                    "return_full_text": False,
                    "temperature": temperature,
                    "do_sample": True,
                }
                return self.model(encoded, **temporal_args)[0]['generated_text']
            return self.model.generate(
                encoded,
                max_new_tokens=1024,
                do_sample=True,
                temperature=temperature,
                use_cache=True
            )
        else:
            raise Exception('generation not defined')

    def call(self, message, generation_args):
        if self.model_arg in ['solar', 'commandr']:
            prompt = self.tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            encoded = self.tokenizer(prompt, return_tensors="pt").to(device)
            generated_ids = self.generate_by_arg(encoded, generation_args)
            generated_ids = self.tokenizer.batch_decode(generated_ids)[0]
        elif self.model_arg in ['phi3']:
            generated_ids = self.generate_by_arg(message, generation_args)
        else:
            encoded = self.tokenizer.apply_chat_template(message, return_tensors="pt").to(device)
            generated_ids = self.generate_by_arg(encoded, generation_args)
            generated_ids = self.tokenizer.batch_decode(generated_ids)[0]

        return generated_ids


def run_evaluation(
        client,
        template_type,
        option_type,
        trial,
        shot,
        o_dir,
        eval_model,
        generation_args
):
    _data = read_jsonl("../Data/ioinst.jsonl")
    _data, _messages = apply_template(_data, template_type=template_type, option_type=option_type, shot=shot)

    # ceate output folder if not exists
    _o_dir = join(o_dir, option_type, eval_model)
    os.makedirs(_o_dir, exist_ok=True)

    _opath = join(_o_dir, f"results_template{template_type}_{generation_args}_trial{trial}.jsonl")


    # load_results if exists
    if os.path.exists(_opath):
        _exist = load_jsonl(_opath)
        _exist_ids = [i['id'] for i in _exist]
        for pos, instance in enumerate(_data):
            if instance['id'] in _exist_ids:
                _data[pos] = _exist[_exist_ids.index(instance['id'])]

    result_writer = open(_opath, 'w')

    print(f"--------Evaluating template {template_type}--------")
    print(f"--------Evaluation Using {eval_model}--------")
    for entry, message in tqdm(zip(_data, _messages)):
        # skip if eval exists
        if entry.get('generated', None) is not None:
            result_writer.write(json.dumps(entry) + '\n')
            result_writer.flush()
            continue

        success = False
        while not success:
            generation = client.call(
                message=message,
                generation_args=generation_args,
            )
            entry['generated'] = generation
            success = True

        torch.cuda.empty_cache()

        result_writer.write(json.dumps(entry) + '\n')
        result_writer.flush()

    result_writer.close()
    return _opath


def main_run(args, client):
    directory = args.output_dir if args.shot == 0 else args.output_dir + f'_shot{args.shot}'

    run_evaluation(
        client=client,
        template_type=args.template_type,
        option_type=args.option_type,
        trial=args.trial,
        shot=args.shot,
        o_dir=directory,
        eval_model=args.model,
        generation_args=args.generation_args
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="mistral", help="model name to be used for evaluation")
    parser.add_argument("--template_type", default="random", help="")
    parser.add_argument("--option_type", type=str, default="hard", help=" easy, hard, veryhard")
    parser.add_argument("--shot", type=int, default=0, help="shot")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./eval_results', help="path to the output folder")
    parser.add_argument("--generation_args", type=str, default="greedy")
    temporal_args = parser.parse_args()

    model, tokenizer = hf_call(temporal_args.model)
    client = HF(model, tokenizer, temporal_args.model)

    trials = [0, 1, 2, 3, 4]
    options = ["easy", "hard", "veryhard"]
    # template_types = ["random", "long", "short", "firstop", "firstcd"]
    template_types = ["random"]
    shots = [0, 1, 3]
    for trial in trials:
        for option in options:
            for template_type in template_types:
                for shot in shots:
                    print(f"now: {option}-{template_type}-shot{shot}{trial}")
                    temporal_args.template_type = template_type
                    temporal_args.option_type = option
                    temporal_args.shot = shot
                    temporal_args.trial = trial
                    main_run(temporal_args, client)

