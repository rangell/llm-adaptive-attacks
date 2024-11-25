import copy
import pickle
from datasets import load_dataset

from IPython import embed


# Types of jailbreaks:
JAILBREAK_PROMPT_NAMES = ['AIM',
                          'autodan',
                          'base64',
                          'base64_input_only',
                          'base64_output_only',
                          'base64_raw',
                          'combination_1',
                          'combination_2',
                          'combination_3',
                          'dev_mode_v2',
                          'disemvowel',
                          'distractors',
                          'distractors_negated',
                          'evil_confidant',
                          'evil_system_prompt',
                          'few_shot_json',
                          'gcg',
                          'leetspeak',
                          'low_resource',
                          'low_resource_english',
                          'multi_shot_25',
                          'multi_shot_5',
                          'none',
                          'obfuscation',
                          'original_prompt',
                          'poems',
                          'prefix_injection',
                          'prefix_injection_hello',
                          'refusal_suppression',
                          'refusal_suppression_inv',
                          'rot13',
                          'style_injection_json',
                          'style_injection_short',
                          'wikipedia',
                          'wikipedia_with_title']


def multi_shot_prompt_to_convo(prompt):
    turn_segments = [s.replace("<end_of_turn>\n", "") for s in prompt.split("<start_of_turn>") if s != "model\n"]
    multi_shot_convo = []
    for segment in turn_segments:
        if segment[:5] == "user\n":
            multi_shot_convo.append({"role": "user", "content": segment[5:]})
        elif segment[:6] == "model\n":
            multi_shot_convo.append({"role": "assistant", "content": segment[6:]})
        elif segment == "<bos>":
            continue
        else:
            print("Error: ", segment)
    return multi_shot_convo


def prompt_to_convo(prompt):
    return [{"role": "user", "content": prompt}]


if __name__ == "__main__":
    jailbreak_dataset = load_dataset("sevdeawesome/jailbreak_success")

    augmented_examples = []
    for x in jailbreak_dataset["train"]:
        # remove duplicates (these are duplicates of "original_prompt")
        if x["jailbreak_prompt_name"] in ["none", "evil_system_prompt"]:
            continue

        # generate the convo
        _x = copy.deepcopy(x)
        if _x["jailbreak_prompt_name"].find("multi_shot") != -1:
            _x["jailbreak_prompt_convo"] = multi_shot_prompt_to_convo(_x["jailbreak_prompt_text"])
        else:
            _x["jailbreak_prompt_convo"] = prompt_to_convo(_x["jailbreak_prompt_text"])
        augmented_examples.append(_x)

    with open("data/jailbreak_success_preprocessed.pkl", "wb") as f:
        pickle.dump(augmented_examples, f)
