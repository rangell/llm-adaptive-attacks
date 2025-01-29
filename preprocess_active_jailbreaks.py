from collections import defaultdict, Counter
import glob
import os
import pickle

from IPython import embed


if __name__ == "__main__":
    dumps_dir = "/scratch/rca9780/llm-adaptive-attacks-data/jailbreak_dumps/"

    dumps_map = defaultdict(list)
    for name in glob.glob(dumps_dir + "*.pkl"):
        modified_name = os.path.basename(name).removesuffix(".pkl")
        model, instruction = modified_name.split("--")
        if instruction != "goal":
            dumps_map[model].append(name)

    model_names = []
    for model, paths in dumps_map.items():
        if len(paths) == 50:
            model_names.append(model)
            
    jailbreak_data = defaultdict(list)
    for model in model_names:
        for dump_path in dumps_map[model]:
            instruction = os.path.basename(name).removesuffix(".pkl").split("--")[1]
            with open(dump_path, "rb") as f:
                local_data = pickle.load(f)
            jailbreak_data["source_model"].append(model)
            jailbreak_data["instruction"].append(instruction)
            jailbreak_data["source_prob"].append(local_data["probs"][-1])
            jailbreak_data["source_strength"].append(local_data["jailbreak_strengths"][-1])
            jailbreak_data["input_sequence"].append(local_data["input_sequences"][-1])
    
    with open("data/active_jailbreaks.pkl", "wb") as f:
        pickle.dump(jailbreak_data, f)