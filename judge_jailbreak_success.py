import argparse
from argparse import Namespace
from collections import defaultdict
import copy
import os
import json
import itertools
import random
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from conversers import load_target_model
from judges_ensemble import load_judge, judge_rule_based


from IPython import embed


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-model",
        default="llama3-8b",     # see conversers.py for a whole list of target models
        help="Name of target model."
    )
    parser.add_argument(
        "--jailbreak-dataset",
        default="jailbreak_success",
        help="Jailbreak dataset to use.",
        choices=["jailbreak_success", "wildjailbreak"]
    )

    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    target_lm = load_target_model(Namespace(**{"target_model": args.target_model}))

    judge_name = "gpt-4-0613"
    #judge_name = "gpt-4-1106-preview" 
    #judge_name = "gpt-4o-2024-08-06" 
    judge_max_n_tokens = 10
    judge_temperature = 0.0
    judge_top_p = 1.0
    judge_lm = load_judge(Namespace(**{"judge_model": judge_name,
                                       "judge_max_n_tokens": judge_max_n_tokens,
                                       "judge_temperature": judge_temperature,
                                       "judge_top_p": judge_top_p}))

    # first pass: simple rule judge
    target_max_n_tokens = 10
    temperature = 1.0
    batch_size = 32

    if args.jailbreak_dataset == "jailbreak_success":
        jailbreak_dataset = load_dataset("sevdeawesome/jailbreak_success")
        jailbreak_dataloader = DataLoader(jailbreak_dataset['train'], batch_size=batch_size, shuffle=False)
        prompt_key = "jailbreak_prompt_text"
    elif args.jailbreak_dataset == "wildjailbreak":
        assert False
        wildjailbreak_dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
        adv_harmful_subset = [x for x in wildjailbreak_dataset['train'] if x['data_type'] == 'adversarial_harmful']
        jailbreak_dataloader = DataLoader(adv_harmful_subset, batch_size=batch_size, shuffle=False)
        prompt_key = "adversarial"

    #skipped_batches = 0
    #batch_responses = []
    #for batch in tqdm(jailbreak_dataloader):
    #    try:
    #        _batch = copy.deepcopy(batch)
    #        responses = target_lm.get_response(batch[prompt_key], max_n_tokens=target_max_n_tokens, temperature=temperature, text_only=True)
    #        _batch['responses'] = [resp['text'] for resp in responses]
    #        _batch['jailbroken_rule_judge'] = [judge_rule_based(resp) for resp in _batch['responses']]
    #        batch_responses.append(_batch)
    #    except:
    #        skipped_batches += 1
    #        print("skipping batch: ", _batch)
    #        continue

    #print("number of skipped batches: ", skipped_batches)

    #with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-responses.pkl', 'wb') as f:
    #    pickle.dump(batch_responses, f)

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-responses.pkl', 'rb') as f:
        batch_responses = pickle.load(f)

    # filtering batch_responses for only jailbroken ones
    prompt_responses = {}
    filtered_prompt_responses = []
    for batch in batch_responses:
        for i in range(len(batch["prompt_name"])):
            example_dict = {}
            for key in batch.keys():
                example_dict[key] = batch[key][i]
            example_dict['jailbroken_llm_judge'] = False
            prompt_responses[batch["jailbreak_prompt_text"][i]] = example_dict
            if example_dict['jailbroken_rule_judge'] == True:
                filtered_prompt_responses.append(example_dict)

    target_max_n_tokens = 150
    temperature = 1.0
    batch_size = 32

    batch_responses = []
    filtered_jailbreak_dataloader = DataLoader(filtered_prompt_responses, batch_size=batch_size, shuffle=False)

    score_llm_judge = lambda goal, resp : (judge_lm.score([goal], [resp])[0] == 10)

    # second pass: use an LLM to judge a longer response
    for batch in tqdm(filtered_jailbreak_dataloader):
        _batch = copy.deepcopy(batch)
        responses = target_lm.get_response(batch[prompt_key], max_n_tokens=target_max_n_tokens, temperature=temperature, text_only=True)
        _batch['responses'] = [resp['text'] for resp in responses]
        _batch['jailbroken_rule_judge'] = [judge_rule_based(resp) for resp in _batch['responses']]
        try:
            _batch['jailbroken_llm_judge'] = Parallel(n_jobs=batch_size, backend="threading")(delayed(score_llm_judge)(goal, resp)
                                                                                            for goal, resp in zip(_batch['jailbreak_prompt_text'], _batch['responses']))
        except:
            # just default to saying False
            _batch['jailbroken_llm_judge'] = [False for resp in _batch['responses']]

        batch_responses.append(_batch)

    for batch in batch_responses:
        for i in range(len(batch["prompt_name"])):
            example_dict = {}
            for key in batch.keys():
                example_dict[key] = batch[key][i]
            prompt_responses[batch["jailbreak_prompt_text"][i]] = example_dict

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-filtered_responses.pkl', 'wb') as f:
        pickle.dump(prompt_responses, f)