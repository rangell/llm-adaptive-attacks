import common
import torch
import os
from typing import List
from language_models import GPT, HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from config import *

from IPython import embed


def load_target_model(args):
    targetLM = TargetLM(model_name = args.target_model, 
                        temperature = TARGET_TEMP, # init to 0
                        top_p = TARGET_TOP_P, # init to 1
                        )
    return targetLM

class TargetLM():
    """
    Base class for target language models.
    
    Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """
    def __init__(self, 
            model_name: str, 
            temperature: float,
            top_p: float):
        
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        assert hasattr(self.model.tokenizer, "chat_template")
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0

    def get_full_prompts(self, prompts_list: List[str]) -> List[str]:
        full_prompts = []
        for prompt in prompts_list:
            chat = [{"role": "user", "content": prompt}]
            full_prompts.append(self.model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True))
        return full_prompts

    def get_response(self, prompts_list: List[str], max_n_tokens=None, temperature=None, text_only=False, apply_chat_template=True) -> List[dict]:
        if apply_chat_template:
            full_prompts = self.get_full_prompts(prompts_list)
        else:
            full_prompts = prompts_list

        outputs = self.model.generate(full_prompts, 
                                      max_n_tokens=max_n_tokens,  
                                      temperature=self.temperature if temperature is None else temperature,
                                      text_only=text_only,
                                      top_p=self.top_p,
                                      skip_special_tokens=True
        )
        self.n_input_tokens += sum(output['n_input_tokens'] for output in outputs)
        self.n_output_tokens += sum(output['n_output_tokens'] for output in outputs)
        self.n_input_chars += sum(len(full_prompt) for full_prompt in full_prompts)
        self.n_output_chars += len([len(output['text']) for output in outputs])
        return outputs


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)
    
    if 'gpt' in model_name or 'together' in model_name:
        lm = GPT(model_name)
    else:
        config = AutoConfig.from_pretrained(model_path, output_hidden_states=True)
        model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                #force_download=True,
                token=os.getenv("HF_TOKEN"),
                config=config,
                trust_remote_code=True).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=False,
            token=os.getenv("HF_TOKEN"),
            padding_side="left"
        )

        if 'llama2' in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = 'left'
        if 'vicuna' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
        if 'mistral' in model_path.lower() or 'mixtral' in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)
    
    return lm, template

def get_model_path_and_template(model_name):
    # TODO: remove "template"???
    full_model_dict={
        "gpt-4-0125-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4-1106-preview":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-4":{
            "path":"gpt-4",
            "template":"gpt-4"
        },
        "gpt-3.5-turbo": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "gpt-3.5-turbo-1106": {
            "path":"gpt-3.5-turbo",
            "template":"gpt-3.5-turbo"
        },
        "vicuna":{
            "path":VICUNA_PATH,
            "template":"vicuna_v1.1"
        },
        "llama2":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2"
        },
        "llama2-7b":{
            "path":LLAMA_7B_PATH,
            "template":"llama-2"
        },
        "llama2-13b":{
            "path":LLAMA_13B_PATH,
            "template":"llama-2"
        },
        "llama2-70b":{
            "path":LLAMA_70B_PATH,
            "template":"llama-2"
        },
        "llama3-8b":{
            "path":LLAMA3_8B_PATH,
            "template":"llama-2"
        },
        "llama3-70b":{
            "path":LLAMA3_70B_PATH,
            "template":"llama-2"
        },
        "llama3.1-8b":{
            "path":LLAMA3p1_8B_PATH,
            "template":"llama-2"
        },
        "llama3.1-70b":{
            "path":LLAMA3p1_70B_PATH,
            "template":"llama-2"
        },
        "llama3.2-1b":{
            "path":LLAMA3p2_1B_PATH,
            "template":"llama-2"
        },
        "llama3.2-3b":{
            "path":LLAMA3p2_3B_PATH,
            "template":"llama-2"
        },
        "gemma-2b":{
            "path":GEMMA_2B_PATH,
            "template":"gemma"
        },
        "gemma-7b":{
            "path":GEMMA_7B_PATH,
            "template":"gemma"
        },
        "gemma1.1-2b":{
            "path":GEMMA1p1_2B_PATH,
            "template":"gemma"
        },
        "gemma1.1-7b":{
            "path":GEMMA1p1_7B_PATH,
            "template":"gemma"
        },
        "gemma2-2b":{
            "path":GEMMA2_2B_PATH,
            "template":"gemma"
        },
        "gemma2-9b":{
            "path":GEMMA2_9B_PATH,
            "template":"gemma"
        },
        "gemma2-27b":{
            "path":GEMMA2_27B_PATH,
            "template":"gemma"
        },
        "qwen2.5-0.5b":{
            "path":QWEN2p5_0p5B_PATH,
            "template":"qwen"
        },
        "qwen2.5-1.5b":{
            "path":QWEN2p5_1p5B_PATH,
            "template":"qwen"
        },
        "qwen2.5-3b":{
            "path":QWEN2p5_3B_PATH,
            "template":"qwen"
        },
        "qwen2.5-7b":{
            "path":QWEN2p5_7B_PATH,
            "template":"qwen"
        },
        "qwen2.5-14b":{
            "path":QWEN2p5_14B_PATH,
            "template":"qwen"
        },
        "qwen2.5-32b":{
            "path":QWEN2p5_32B_PATH,
            "template":"qwen"
        },
        "mistral-7b":{
            "path":MISTRAL_7B_PATH,
            "template":"mistral"
        },
        "mixtral-7b":{
            "path":MIXTRAL_7B_PATH,
            "template":"mistral"
        },
        "r2d2":{
            "path":R2D2_PATH,
            "template":"zephyr"
        },
        "phi3":{
            "path":PHI3_MINI_PATH,
            "template":"llama-2"  # not used
        },
        "claude-instant-1":{
            "path":"claude-instant-1",
            "template":"claude-instant-1"
        },
        "claude-2":{
            "path":"claude-2",
            "template":"claude-2"
        },
        "palm-2":{
            "path":"palm-2",
            "template":"palm-2"
        }
    }
    # template = full_model_dict[model_name]["template"] if model_name in full_model_dict else "gpt-4"
    assert model_name in full_model_dict, f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template


    
