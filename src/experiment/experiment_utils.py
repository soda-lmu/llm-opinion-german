import re
import os
import json
import torch
from transformers import  BitsAndBytesConfig
from src.paths import PROMPT_DIR

def llama2_remove_tag(output_decoded):
    return re.sub(r".*\[INST\]\s*", "", output_decoded, flags=re.DOTALL)

def gemma_remove_tag(output_decoded):
    split_output = output_decoded.split("model\n", 1)
    without_tags = split_output[1] if len(split_output) > 1 else ""
    return without_tags

def mixtral_remove_tag(output_decoded):
    return re.sub(r".*\[/INST\]\s*", "", output_decoded)

def llama3_remove_tag(output_decoded):
    pattern = r"(?<=assistant\n).+"
    result = re.search(pattern, output_decoded, re.DOTALL).group() # type: ignore
    return result


def get_experiment_config(fpath):

    with open(fpath) as f:
        config = json.load(f)

    config['prompt_fpath']=os.path.join(PROMPT_DIR,config['prompt_fname'])  

    model_name = config['model_name']
    if config['device'] == 'cuda':
        if 'bnb_4bit_compute_dtype' in config['quantization_config']:
            dtype_str = config['quantization_config']['bnb_4bit_compute_dtype']
            if dtype_str == 'float16':
                dtype = torch.float16
            elif dtype_str == 'float32':
                dtype = torch.float32
            
            config['quantization_config']['bnb_4bit_compute_dtype'] = dtype
        
        config['quantization_config'] = BitsAndBytesConfig(**config['quantization_config'])
    if 'Mixtral' in model_name:
        config['remove_tag_fnc'] = mixtral_remove_tag
    elif 'Llama-2-13b-chat-hf' in model_name:
        config['remove_tag_fnc'] = llama2_remove_tag
    elif 'Meta-Llama-3.1' in model_name:
        config['remove_tag_fnc'] = llama3_remove_tag
    elif 'gemma' in model_name:
        config['remove_tag_fnc'] = gemma_remove_tag

    return config
