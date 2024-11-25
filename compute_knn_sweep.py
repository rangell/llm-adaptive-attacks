import argparse
import itertools
import json
import os
import tempfile


SBATCH_TEMPLATE = """
#!/bin/bash
#
#SBATCH --job-name=__job_name__
#SBATCH --output=__out_path__.out
#SBATCH -e __out_path__.err
#
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c 'source /ext3/env.sh; python -u compute_knn_graphs.py --target-model __model_name__ --instructions-dataset __jailbreak_dataset__'\
"""


if __name__ == "__main__":

    models = [
        #"llama2-7b",
        #"llama3-8b",
        "llama3-70b",
        #"llama3.1-8b",
        "llama3.1-70b",
        #"llama3.2-1b",
        #"llama3.2-3b",
        #"gemma-2b",
        #"gemma-7b",
        #"gemma1.1-2b",
        #"gemma1.1-7b",
        #"gemma2-2b",
        #"gemma2-9b",
        "gemma2-27b",
        #"qwen2.5-0.5b",
        #"qwen2.5-1.5b",
        #"qwen2.5-3b",
        #"qwen2.5-7b",
        #"qwen2.5-14b",
        "qwen2.5-32b"
    ]
    #jailbreak_datasets = ["wildjailbreak", "jailbreak_success"]
    jailbreak_datasets = ["wildjailbreak"]

    for model_name in models:
        for jailbreak_dataset in jailbreak_datasets:
            job_name = "{}".format(model_name)
            out_path = "knn_graphs/{}".format(model_name)

            sbatch_str = SBATCH_TEMPLATE.replace("__job_name__", job_name)
            sbatch_str = sbatch_str.replace("__out_path__", out_path)
            sbatch_str = sbatch_str.replace("__model_name__", model_name)
            sbatch_str = sbatch_str.replace("__jailbreak_dataset__", jailbreak_dataset)
            
            print(f"cmd: {model_name}\n")
            with tempfile.NamedTemporaryFile() as f:
                f.write(bytes(sbatch_str.strip(), "utf-8"))
                f.seek(0)
                os.system(f"sbatch {f.name}")
