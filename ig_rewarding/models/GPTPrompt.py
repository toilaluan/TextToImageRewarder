from transformers import pipeline, AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import torch
from urllib.request import urlretrieve
import pandas as pd
import random
from typing import List
from ig_rewarding.models.prompt_expansion import FooocusExpansion


class GPTPrompt(nn.Module):
    def __init__(self, model_name, device, fooocus_cfg, use_fooocus=True):
        super().__init__()
        self.use_fooocus = use_fooocus
        if use_fooocus:
            self.fooocus = FooocusExpansion(**fooocus_cfg)
        self.prompter = pipeline(
            "text-generation", model=model_name, tokenizer=model_name, device=device
        )

    def clean_prompt(self, prompt):
        # prompt is a string of words, separated by spaces or commas
        # we want to remove all extra commas and extra spaces
        for _ in range(3):
            prompt = prompt.replace(",,", ",")
            prompt = prompt.replace("  ", " ")
        return prompt

    @torch.inference_mode()
    def generate_prompt(self, prefix_prompts: List[str], n_prompts: int = 100):
        outputs = self.prompter(
            prefix_prompts, max_length=28, num_return_sequences=n_prompts
        )
        prompt_sets = []
        for prompt_set in outputs:
            prompts = [prompt["generated_text"] for prompt in prompt_set]

            if self.use_fooocus:
                prompts = [
                    self.clean_prompt(self.fooocus(prompt)) for prompt in tqdm(prompts)
                ]
            prompt_sets.append(prompts)
        return prompt_sets


if __name__ == "__main__":
    prompter = GPTPrompt(
        model_name="Gustavosta/MagicPrompt-Stable-Diffusion",
        device="cuda",
        fooocus_cfg={
            "url": "https://huggingface.co/lllyasviel/misc/resolve/main/fooocus_expansion.bin",
            "device": "cuda",
        },
    )
    prompts = prompter.generate_prompt(["a realistic image of"], 100)
