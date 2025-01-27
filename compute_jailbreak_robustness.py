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
from scipy.stats import multivariate_normal
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import seaborn as sns
from tqdm import tqdm

from repe import repe_pipeline_registry # register 'rep-reading' and 'rep-control' tasks into Hugging Face pipelines

from conversers import load_target_model
from judges_ensemble import load_judge, judge_rule_based


def reformat_reps(orig_reps):
    _per_layer_dict = {}
    for k in orig_reps[0].keys():
        _per_layer_dict[k] = torch.concat([x[k] for x in orig_reps])
    out_reps = _per_layer_dict
    for k, reps in out_reps.items():
        out_reps[k] = reps.numpy()
    return out_reps


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
    parser.add_argument(
        "--output-dir",
        help="Path to output directory.",
        required=True
    )
    args = parser.parse_args()

    print("args: ", vars(args))

    # for determinism, maybe need more?
    random.seed(42)

    # register RepE pipelines
    repe_pipeline_registry()

    target_lm = load_target_model(Namespace(**{"target_model": args.target_model}))
    
    # load harmless and harmful datasets
    alpaca_dataset = load_dataset("tatsu-lab/alpaca")
    advbench_dataset = load_dataset("walledai/AdvBench")

    harmless_instructions = [x["instruction"] for x in alpaca_dataset["train"]]
    harmful_instructions = [x["prompt"] for x in advbench_dataset["train"]]

    random.shuffle(harmless_instructions)
    random.shuffle(harmful_instructions)

    orig_harmless_instructions = copy.deepcopy(harmless_instructions)

    harmless_instructions = harmless_instructions[:500]
    harmful_instructions = harmful_instructions[:500]

    del advbench_dataset
    del alpaca_dataset

    # Compute refusal feature direction
    hidden_layers = list(range(-1, -(target_lm.model.model.config.num_hidden_layers), -1))
    #hidden_layers = [-1]
    rep_token = -1
    rep_reading_pipeline =  pipeline("rep-reading", model=target_lm.model.model, tokenizer=target_lm.model.tokenizer)

    harmless_prompts = target_lm.get_full_prompts(harmless_instructions)
    harmful_prompts = target_lm.get_full_prompts(harmful_instructions)

    harmless_reps = rep_reading_pipeline(harmless_prompts, rep_token=rep_token, hidden_layers=hidden_layers, batch_size=128)
    harmless_reps = reformat_reps(harmless_reps)

    harmful_reps = rep_reading_pipeline(harmful_prompts, rep_token=rep_token, hidden_layers=hidden_layers, batch_size=128)
    harmful_reps = reformat_reps(harmful_reps)

    refusal_directions = {}
    for layer_idx in harmless_reps.keys():
        _mean_harmless = np.mean(harmless_reps[layer_idx], axis=0)
        _mean_harmful = np.mean(harmful_reps[layer_idx], axis=0)
        _refusal_direction = _mean_harmful - _mean_harmless
        _refusal_direction /= np.linalg.norm(_refusal_direction)
        refusal_directions[layer_idx] = _refusal_direction

    with open(f"data/{args.jailbreak_dataset}_preprocessed.pkl", "rb") as f:
        _jailbreak_dataset = pickle.load(f)
    sized_examples = []
    for x in _jailbreak_dataset:
        x["jailbreak_prompt_text"] = target_lm.model.tokenizer.apply_chat_template(x["jailbreak_prompt_convo"], tokenize=False, add_generation_prompt=True)
        num_tokens = len(target_lm.model.tokenizer(x["jailbreak_prompt_text"])['input_ids'])
        sized_examples.append((num_tokens, x))
    sized_examples = sorted(sized_examples, key=lambda x : x[0])
    jailbreak_dataset = [x[1] for x in sized_examples]

    original_instructions = [x["original_prompt_text"] for x in jailbreak_dataset]
    original_prompts = target_lm.get_full_prompts(original_instructions)
    jailbreak_prompts = [x["jailbreak_prompt_text"] for x in jailbreak_dataset]

    print("Original reps...")
    original_reps = rep_reading_pipeline(original_prompts, rep_token=rep_token, hidden_layers=hidden_layers, batch_size=16)
    original_reps = reformat_reps(original_reps)
    print("Done.")

    print("Jailbreak reps...")
    jailbreak_reps = rep_reading_pipeline(jailbreak_prompts, rep_token=rep_token, hidden_layers=hidden_layers, batch_size=1)
    jailbreak_reps = reformat_reps(jailbreak_reps)
    print("Done.")

    jailbreak_robustness_data = {
        "target_model": args.target_model,
        "jailbreak_dataset_name": args.jailbreak_dataset,
        "jailbreak_dataset": jailbreak_dataset,
        "harmful_prompts": harmful_prompts,
        "harmful_reps": harmful_reps,
        "harmless_prompts": harmless_prompts,
        "harmless_reps": harmless_reps,
        "original_prompts": original_prompts,
        "original_reps": original_reps,
        "jailbreak_prompts": jailbreak_prompts,
        "jailbreak_reps": jailbreak_reps,
    }

    with open(f'{args.output_dir}/{args.target_model}-{args.jailbreak_dataset}-robustness_data.pkl', 'wb') as f:
        pickle.dump(jailbreak_robustness_data, f)