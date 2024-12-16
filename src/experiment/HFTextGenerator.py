import os
import time
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HUGGINGFACE_HUB_CACHE = (
    os.environ.get("HUGGINGFACE_HUB_CACHE")
    if os.environ.get("HUGGINGFACE_HUB_CACHE") != ""
    else None
)

logger = logging.getLogger(__name__)

class HFTextGenerator:
    def __init__(self, model_name, device, quantization_config):
        self.tokenizer, self.model, self.device = self.load_model(
            model_name, device, quantization_config
        )

    def load_model(self, model_name, device, quantization_config):
        logger.info(f"Loading model: {model_name} with device: {device}")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=HUGGINGFACE_HUB_CACHE, device=device, padding_side="left"
        )
        if quantization_config is not None or quantization_config != {}:
            logger.info("Using quantization configuration {} for model".format(quantization_config))
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=HUGGINGFACE_HUB_CACHE,
                quantization_config=quantization_config
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, cache_dir=HUGGINGFACE_HUB_CACHE
            )
        model.name = model_name
        logger.info(f"Model {model_name} loaded successfully")
        return tokenizer, model, device

    def generate_response(self, prompt, generation_config, remove_tag):
        begin = time.time()

        model_input = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_num_token = model_input.input_ids.size(1)
        with torch.no_grad():
            output = self.model.generate(**model_input, **generation_config)[0]
            output_decoded = self.tokenizer.decode(output, skip_special_tokens=True)
        end = time.time()
        d = {
            "model": self.model.name,
            "prompt": prompt,
            "output": remove_tag(
                output_decoded
            ),
        }
        d.update(generation_config)
        return d


    def generate_batch_response(self, prompt_list, generation_config, remove_tag):
        self.tokenizer.pad_token = self.tokenizer.eos_token
        model_inputs = self.tokenizer(prompt_list, return_tensors="pt", padding=True).to(self.device) 
        with torch.no_grad():
            outputs = self.model.generate(**model_inputs, **generation_config)
            outputs_decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            outputs_decoded_cleared = [remove_tag(output_decoded) for output_decoded in outputs_decoded]

        return outputs_decoded_cleared

def main():
    pass


if __name__ == "__main__":
    main()
