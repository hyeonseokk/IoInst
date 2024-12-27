import json
import os
import time
# import tiktoken
import argparse

from os.path import join,exists
from openai import OpenAI
from tqdm import tqdm
from utils import load_jsonl, read_jsonl, apply_template


def run_evaluation(
        client,
        template_type,
        option_type,
        trial,
        shot,
        o_dir,
        eval_model,
        temperature
):

    _data = read_jsonl("../Data/ioinst.jsonl")
    _data, _messages = apply_template(_data, template_type=template_type, option_type=option_type, shot=shot)

    # ceate output folder if not exists
    _o_dir = join(o_dir, option_type, eval_model)
    os.makedirs(_o_dir, exist_ok=True)

    _opath = join(_o_dir, f"results_template{template_type}_trial{trial}.jsonl")

    # load_results if exists
    if os.path.exists(_opath):
        _exist = load_jsonl(_opath)
        _exist_ids = [i['id'] for i in _exist]
        for pos, instance in enumerate(_data):
            if instance['id'] in _exist_ids:
                _data[pos] = _exist[_exist_ids.index(instance['id'])]

    result_writer = open(_opath, 'w')

    print(f"--------Evaluating template {template_type} - {option_type}--------")
    print(f"--------Evaluation Using {eval_model}--------")
    for entry, message in tqdm(zip(_data, _messages)):
        # skip if eval exists
        if entry.get('generated', None) is not None:
            result_writer.write(json.dumps(entry) + '\n')
            result_writer.flush()
            continue

        # print(f"--------Instance {entry['id']}--------")
        success = False
        while not success:
            try:
                completion = client.chat.completions.create(
                    model=eval_model,
                    messages=message,
                    temperature=temperature,
                )
                generation = completion.choices[0].message.content
                entry['generated'] = generation
                # entry['options'] = entry[f"options_{option_type}"]
                # entry['label'] = entry[f"label_{option_type}"]
                success = True
            except Exception as e:
                print("ERROR!")
                print(e)
                print("Retry!")
                time.sleep(20)

        result_writer.write(json.dumps(entry) + '\n')
        result_writer.flush()

    result_writer.close()
    return _opath


def main_run(args):
    client = OpenAI(api_key=args.api_key)

    directory = args.output_dir if args.shot == 0 else args.output_dir + f'_shot{args.shot}'

    run_evaluation(
        client=client,
        template_type=args.template_type,
        option_type=args.option_type,
        o_dir=directory,
        trial=args.trial,
        shot=args.shot,
        eval_model=args.model,
        temperature=args.temperature
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="openai_api_key")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo-16k", choices=[
        "gpt-3.5-turbo-16k", "gpt-4o", "gpt-4o-mini"
    ])
    parser.add_argument("--template_type", default="random", help="")
    parser.add_argument("--option_type", type=str, default="veryhard", help="easy, hard, veryhard")
    parser.add_argument("--shot", type=int, default=0, help="shot")
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default='./eval_results', help="path to the output folder")
    parser.add_argument("--temperature", type=float, default=0, help="temperature to be used for evaluation")
    tmporal_args = parser.parse_args()

    # for trial in [0, 1, 2, 3, 4]:
    for trial in [0]:
        for shot in [1]:
            for option in ['veryhard', 'hard', 'easy']:
                tmporal_args.option_type = option
                tmporal_args.trial = trial
                tmporal_args.shot = shot
                print(f"option: {tmporal_args.option_type}")
                print(f"trial: {tmporal_args.trial}")
                print(f"shot: {tmporal_args.shot}")
                main_run(tmporal_args)
