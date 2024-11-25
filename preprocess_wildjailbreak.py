import copy
import pickle
from datasets import load_dataset

from IPython import embed


def prompt_to_convo(prompt):
    return [{"role": "user", "content": prompt}]


if __name__ == "__main__":
    jailbreak_dataset = load_dataset("allenai/wildjailbreak", "train", delimiter="\t", keep_default_na=False)

    augmented_examples = []
    for x in jailbreak_dataset["train"]:
        # remove duplicates (these are duplicates of "original_prompt")
        if x["data_type"] != "adversarial_harmful":
            continue
        
        # reformat wildjailbreak examples to the same format as "sevdeawesome/jailbreak_success"
        _x = {
            "prompt_name": "",
            "jailbreak_prompt_name": "wildjailbreak/train/adversarial_harmful",
            "jailbreak_prompt_text": x["adversarial"],
            "original_prompt_text": x["vanilla"]
        }
        # generate the convo
        _x["jailbreak_prompt_convo"] = prompt_to_convo(_x["jailbreak_prompt_text"])

        augmented_examples.append(_x)

    with open("data/wildjailbreak_preprocessed.pkl", "wb") as f:
        pickle.dump(augmented_examples, f)
