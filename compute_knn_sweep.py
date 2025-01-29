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
#SBATCH --gres=gpu:rtx8000:__num_gpus__
#SBATCH --mem=64G
#SBATCH --time=0-24:00:00

singularity exec --nv\
            --overlay /scratch/rca9780/jailbreaks/overlay-15GB-500K.ext3:ro \
            /scratch/work/public/singularity/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif /bin/bash \
            -c 'source /ext3/env.sh; python -u compute_knn_graphs.py --target-model __model_name__ --output-dir __output_dir__'\
"""


if __name__ == "__main__":

    models = [
        "llama3-70b",
        "llama3.1-70b",
        #"llama3-8b",
        #"llama3.1-8b",
        #"llama3.2-1b",
        #"llama3.2-3b",
        #"gemma-2b",
        #"gemma-7b",
        #"gemma1.1-2b",
        #"gemma1.1-7b",
        #"gemma2-2b",
        #"gemma2-9b",
        #"gemma2-27b",
        #"qwen2.5-0.5b",
        #"qwen2.5-1.5b",
        #"qwen2.5-3b",
        #"qwen2.5-7b",
        #"qwen2.5-14b",
        #"qwen2.5-32b"
    ]

    output_dir = "/scratch/rca9780/llm-adaptive-attacks-data/knn_graphs/"

    for model_name in models:

        model_size = float(model_name.split("-")[1].replace("b", ""))
        if model_size >= 70:
            num_gpus = "4"
        elif model_size >= 27:
            num_gpus = "2"
        else:
            num_gpus = "1"

        job_name = "{}".format(model_name)
        out_path = "{}/{}".format(output_dir, model_name)

        sbatch_str = SBATCH_TEMPLATE.replace("__job_name__", job_name)
        sbatch_str = sbatch_str.replace("__out_path__", out_path)
        sbatch_str = sbatch_str.replace("__output_dir__", output_dir)
        sbatch_str = sbatch_str.replace("__model_name__", model_name)
        sbatch_str = sbatch_str.replace("__num_gpus__", num_gpus)
            
        print(f"cmd: {model_name}\n")
        with tempfile.NamedTemporaryFile() as f:
            f.write(bytes(sbatch_str.strip(), "utf-8"))
            f.seek(0)
            os.system(f"sbatch {f.name}")
