import os
from os.path import join
import argparse

from torchmetrics.text.rouge import ROUGEScore

from utils import load_jsonl, read_jsonl

ROUGE = ROUGEScore()


def define_correct(hyp, ref):
    rouge = ROUGE(hyp, ref)['rougeL_precision']
    threshold = 0.9
    return rouge > threshold


def truncate_output(string, model):
    if model in ['gpt3.5', 'gpt4']:
        return string
    elif model in ['mistral', 'llama2', 'llama2_13', 'mixtral']:
        return string.split('[/INST]')[1][:-4]
    elif model in ['llama3']:
        return string.split('<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n')[-1]
    elif model in ['solar']:
        return string.split('### Assistant')[-1]
    elif model in ['gemma']:
        return string.split('<end_of_turn>')[1][:-4]
    elif model in ['commandr']:
        return string.split('<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>')[-1].split('<|END_OF_TURN_TOKEN|>')[0]
    else:
        raise Exception('err')


def main(df, model):
    eval_results = {
        'correct_global': [],
        'wrong_global': [],
        'correct_inst': [],
        'wrong_inst': [],
        'duplicates': [],
    }
    for item in df:
        global_ans = False
        inst_ans = False
        duplicates = False
        generated = truncate_output(item['generated'], model)

        for option in item['options']:
            if define_correct(option, generated):
                global_ans = True
        if define_correct(item['options'][item['label']], generated):
            inst_ans = True
            for idx, option in enumerate(item['options']):
                if (idx != item['label']) and (define_correct(option, generated)):
                    duplicates = True

        if global_ans:
            eval_results['correct_global'].append(item['id'])
        else:
            eval_results['wrong_global'].append(item['id'])

        if inst_ans:
            eval_results['correct_inst'].append(item['id'])
        else:
            eval_results['wrong_inst'].append(item['id'])

        if duplicates:
            eval_results['duplicates'].append(item['id'])

    global_acc = len(eval_results['correct_global']) / len(df)
    inst_acc = len(eval_results['correct_inst']) / len(df)

    global_acc = format(global_acc * 100, '.2f')
    inst_acc = format(inst_acc * 100, '.2f')

    results = [global_acc, inst_acc]

    return results


if __name__ == '__main__':
    print(':)')
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt3.5", help="model name to be used for evaluation")
    parser.add_argument("--template_type", default="random", help="")
    parser.add_argument("--option_type", type=str, default="veryhard", help="easy, hard, veryhard")
    parser.add_argument("--trial", type=int, default=0)
    temporal_args = parser.parse_args()

    df = read_jsonl(f'./eval_results/{temporal_args.option_type}/{temporal_args.model}/results_templaterandom_greedy_trial0.jsonl')
    eval_results = main(df, temporal_args.model)

    print(f'ACC1: {eval_results[1]}')
    print(f'ACC2: {eval_results[0]}')

    print(';)')
