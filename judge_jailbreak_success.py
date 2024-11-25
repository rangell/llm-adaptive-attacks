import argparse
from argparse import Namespace
from collections import defaultdict
import copy
import gc
import os
import json
import time
import itertools
import math
import random
import pickle
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader
from joblib import Parallel, delayed

from conversers import load_target_model, get_model_path_and_template
from judges_ensemble import load_judge, judge_rule_based

from IPython import embed


def custom_collate_fn(data):
    batched_data = defaultdict(list)
    for datum in data:
        for k, v in datum.items():
            batched_data[k].append(v)
    return batched_data


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

    # load the target model
    target_lm = load_target_model(Namespace(**{"target_model": args.target_model}))

    # first pass: simple rule judge
    target_max_n_tokens = 10
    temperature = 1.0
    batch_size = 128

    with open(f"data/{args.jailbreak_dataset}_preprocessed.pkl", "rb") as f:
        _jailbreak_dataset = pickle.load(f)
    sized_examples = []
    for x in _jailbreak_dataset:
        x["jailbreak_prompt_text"] = target_lm.model.tokenizer.apply_chat_template(x["jailbreak_prompt_convo"], tokenize=False, add_generation_prompt=True)
        num_tokens = len(target_lm.model.tokenizer(x["jailbreak_prompt_text"])['input_ids'])
        sized_examples.append((num_tokens, x))
    sized_examples = sorted(sized_examples, key=lambda x : x[0])
    jailbreak_dataset = [x[1] for x in sized_examples]
    jailbreak_dataloader = DataLoader(jailbreak_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    prompt_key = "jailbreak_prompt_text"

    subbatch_factor = 1
    batch_responses = []
    for batch in tqdm(jailbreak_dataloader):
        _batch = copy.deepcopy(batch)
        batch_processed = False
        while not batch_processed:
            try:
                responses = []
                subbatch_size = batch_size // subbatch_factor
                batch_length = len(batch["prompt_name"])
                for i in range(math.ceil(batch_length / subbatch_size)):
                    start_idx = i*subbatch_size
                    end_idx = min((i+1)*subbatch_size, batch_length)
                    subbatch_prompts = batch[prompt_key][start_idx:end_idx]
                    subbatch_responses = target_lm.get_response(
                        subbatch_prompts,
                        max_n_tokens=target_max_n_tokens,
                        temperature=temperature,
                        text_only=True,
                        apply_chat_template=False)
                    responses += subbatch_responses
                _batch['response'] = [resp['text'] for resp in responses]
                batch_processed = True
            except torch.OutOfMemoryError as e: 
                gc.collect()
                torch.cuda.empty_cache()
                if subbatch_factor == batch_size:
                    raise ValueError("Subbatch size is already 1!!!")
                subbatch_factor *= 2
                print("new subbatch factor: ", subbatch_factor)
        _batch['jailbroken_rule_judge'] = [judge_rule_based(resp) for resp in _batch['response']]
        batch_responses.append(_batch)

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-short_responses.pkl', 'wb') as f:
        pickle.dump(batch_responses, f)

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-short_responses.pkl', 'rb') as f:
        batch_responses = pickle.load(f)

    # filtering batch_responses for only ones that did not refuse in first 10 tokens
    prompt_responses = {}
    filtered_prompt_responses = []
    for batch in batch_responses:
        for i in range(len(batch["prompt_name"])):
            example_dict = {}
            for key in batch.keys():
                example_dict[key] = batch[key][i]
            prompt_responses[batch["jailbreak_prompt_text"][i]] = example_dict
            if example_dict['jailbroken_rule_judge'] == True:
                filtered_prompt_responses.append(example_dict)

    target_max_n_tokens = 150
    temperature = 1.0
    batch_size = 32

    batch_responses = []
    filtered_jailbreak_dataloader = DataLoader(filtered_prompt_responses, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # second pass: need to generate a longer response to better judge
    subbatch_factor = 1
    batch_responses = []
    for batch in tqdm(filtered_jailbreak_dataloader):
        _batch = copy.deepcopy(batch)
        batch_processed = False
        while not batch_processed:
            try:
                responses = []
                subbatch_size = batch_size // subbatch_factor
                batch_length = len(batch["prompt_name"])
                for i in range(math.ceil(batch_length / subbatch_size)):
                    start_idx = i*subbatch_size
                    end_idx = min((i+1)*subbatch_size, batch_length)
                    subbatch_prompts = batch[prompt_key][start_idx:end_idx]
                    subbatch_responses = target_lm.get_response(
                        subbatch_prompts,
                        max_n_tokens=target_max_n_tokens,
                        temperature=temperature,
                        text_only=True,
                        apply_chat_template=False)
                    responses += subbatch_responses
                _batch['response'] = [resp['text'] for resp in responses]
                batch_processed = True
            except torch.OutOfMemoryError as e: 
                gc.collect()
                torch.cuda.empty_cache()
                if subbatch_factor == batch_size:
                    raise ValueError("Subbatch size is already 1!!!")
                subbatch_factor *= 2
                print("new subbatch factor: ", subbatch_factor)
        _batch['jailbroken_rule_judge'] = [judge_rule_based(resp) for resp in _batch['response']]
        batch_responses.append(_batch)

    for batch in batch_responses:
        for i in range(len(batch["prompt_name"])):
            example_dict = {}
            for key in batch.keys():
                example_dict[key] = batch[key][i]
            prompt_responses[batch["jailbreak_prompt_text"][i]] = example_dict

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-long_responses.pkl', 'wb') as f:
        pickle.dump(prompt_responses, f)

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-long_responses.pkl', 'rb') as f:
        prompt_responses = pickle.load(f)

    # get all of the special tokens, need to remove these from the responses
    # TODO: maybe add skip_special_tokens to generate?
    special_tokens = []
    for _token in target_lm.model.tokenizer.special_tokens_map.values():
        if isinstance(_token, list):
            special_tokens += _token
        else:
            special_tokens.append(_token)

    # load the guard tokenizer 
    guard_model_name = "meta-llama/Llama-Guard-3-8B"
    guard_tokenizer = AutoTokenizer.from_pretrained(guard_model_name, padding_side="left")
    guard_tokenizer.pad_token_id = 0

    # some bookkeeping for the guard model
    augmented_prompt_responses = {}
    for prompt, response in prompt_responses.items():
        _response = copy.deepcopy(response)
        tokenized_response = target_lm.model.tokenizer.tokenize(response["response"])
        tokenized_response = [t for t in tokenized_response if t not in special_tokens]
        detokenized_response = target_lm.model.tokenizer.convert_tokens_to_string(tokenized_response)
        _response["response"] = detokenized_response
        original_chat = [
            {"role": "user", "content": _response["original_prompt_text"]},
            {"role": "assistant", "content": detokenized_response},
        ]
        jailbreak_chat = _response["jailbreak_prompt_convo"] + [{"role": "assistant", "content": detokenized_response}]
        _response["original_conversation"] = guard_tokenizer.apply_chat_template(original_chat, tokenize=False, add_generation_prompt=True)
        _response["jailbreak_conversation"] = guard_tokenizer.apply_chat_template(jailbreak_chat, tokenize=False, add_generation_prompt=True)
        augmented_prompt_responses[prompt] = _response

    # make space for the judge model
    del target_lm
    gc.collect()
    torch.cuda.empty_cache()

    # load guard model
    guard_model = AutoModelForCausalLM.from_pretrained(guard_model_name, torch_dtype=torch.bfloat16, device_map="auto")
    batch_size = 8

    batch_responses = []
    response_dataloader = DataLoader(list(augmented_prompt_responses.values())[-40:], batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
    for batch in tqdm(response_dataloader):
        _batch = copy.deepcopy(batch)
        inputs = guard_tokenizer(_batch["original_conversation"], padding=True, return_tensors="pt").to("cuda")
        output = guard_model.generate(**inputs, max_new_tokens=2, pad_token_id=0)
        prompt_len = inputs["input_ids"].shape[-1]
        _batch["original_guard3_response"] = guard_tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
        _batch["original_guard3_judge"] = [r.find("unsafe") != -1 for r in _batch["original_guard3_response"]]
        inputs = guard_tokenizer(_batch["jailbreak_conversation"], padding=True, return_tensors="pt").to("cuda")
        output = guard_model.generate(**inputs, max_new_tokens=2, pad_token_id=0)
        prompt_len = inputs["input_ids"].shape[-1]
        _batch["jailbreak_guard3_response"] = guard_tokenizer.batch_decode(output[:, prompt_len:], skip_special_tokens=True)
        _batch["jailbreak_guard3_judge"] = [r.find("unsafe") != -1 for r in _batch["jailbreak_guard3_response"]]
        batch_responses.append(_batch)

        # don't really know if this is necessary
        gc.collect()
        torch.cuda.empty_cache()

    prompt_responses = {}
    for batch in batch_responses:
        for i in range(len(batch["prompt_name"])):
            example_dict = {}
            for key in batch.keys():
                example_dict[key] = batch[key][i]
            prompt_responses[batch["jailbreak_prompt_text"][i]] = example_dict

    with open(f'jailbreak_screening_results/{args.jailbreak_dataset}/{args.target_model}-judged_responses.pkl', 'wb') as f:
        pickle.dump(prompt_responses, f)