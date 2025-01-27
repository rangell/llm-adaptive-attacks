from argparse import Namespace
import copy
import random
import os
from datasets import load_dataset
from torch.utils.data import DataLoader
from conversers import load_target_model
from judges_ensemble import judge_rule_based
from tqdm import tqdm
import pickle

# for determinism
random.seed(42)

# login to huggingface for restricted model use
from huggingface_hub import login
login(os.environ["HF_TOKEN"])

# load harmless and harmful datasets
alpaca_dataset = load_dataset("tatsu-lab/alpaca")
advbench_dataset = load_dataset("walledai/AdvBench")

#TODO: fix this!

# wildjailbreak_dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)
# harmless_examples = [x for x in wildjailbreak_dataset["train"] if x["data_type"] == "vanilla_benign"]
# harmful_examples = [x for x in wildjailbreak_dataset["train"] if x["data_type"] == "vanilla_harmful"]

harmless_instructions = [x["instruction"] for x in alpaca_dataset["train"]][:500]
harmful_instructions = [x["prompt"] for x in advbench_dataset["train"]][:500]

# harmless_instructions = [x["vanilla"] for x in harmless_examples]
# harmful_instructions = [x["vanilla"] for x in harmful_examples]

random.shuffle(harmless_instructions)
random.shuffle(harmful_instructions)

orig_harmless_instructions = copy.deepcopy(harmless_instructions)

harmless_instructions = harmless_instructions
harmful_instructions = harmful_instructions


#models = [
#    "llama3-8b",
#    "llama3-70b",
#    "llama3.1-8b",
#    "llama3.1-70b",
#    "llama3.2-1b",
#    "llama3.2-3b",
#    "gemma-2b",
#    "gemma-7b",
#    "gemma1.1-2b",
#    "gemma1.1-7b",
#    "gemma2-2b",
#    "gemma2-9b",
#    "gemma2-27b",
#    "qwen2.5-0.5b",
#    "qwen2.5-1.5b",
#    "qwen2.5-3b",
#    "qwen2.5-7b",
#    "qwen2.5-14b",
#    "qwen2.5-32b"
#]
target_model = "llama3.1-70b"
target_lm = load_target_model(Namespace(**{"target_model": target_model}))

preference_data = []

# first pass: simple rule judge
target_max_n_tokens = 150
temperature = 1.0
batch_size = 32

harmless_dataloader = DataLoader(harmless_instructions, batch_size=batch_size, shuffle=False)
for batch in tqdm(harmless_dataloader):
    batch_responses = target_lm.get_response(
        batch,
        max_n_tokens=target_max_n_tokens,
        temperature=temperature,
        text_only=True,
        apply_chat_template=True)
    batch_responses = [x["text"] for x in batch_responses]
    for prompt, response in zip(batch, batch_responses):
        if judge_rule_based(response):
            preference_data.append({
                "prompt": prompt,
                "conversation": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}],
                "desirable": True
            })

harmful_dataloader = DataLoader(harmful_instructions, batch_size=batch_size, shuffle=False)
for batch in tqdm(harmful_dataloader):
    batch_responses = target_lm.get_response(
        batch,
        max_n_tokens=target_max_n_tokens,
        temperature=temperature,
        text_only=True,
        apply_chat_template=True)
    batch_responses = [x["text"] for x in batch_responses]
    for prompt, response in zip(batch, batch_responses):
        if not judge_rule_based(response):
            preference_data.append({
                "prompt": prompt,
                "conversation": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response}],
                "desirable": False
            })
with open(f"unsafe_preference_data-{target_model}.pkl", "wb") as f:
    pickle.dump(preference_data, f)