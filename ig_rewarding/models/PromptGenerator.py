from transformers import pipeline, AutoModel, AutoTokenizer
import torch.nn as nn
from tqdm import tqdm
import torch
from urllib.request import urlretrieve
import pandas as pd
import random
import chromadb
from typing import List
from chromadb.utils import embedding_functions


class PromptGenerator(nn.Module):
    def __init__(self, chroma_db_config):
        super().__init__()
        self.n_neighbors = chroma_db_config["n_neighbors"]
        self.prompt_generation_pipe = pipeline(
            "text-generation", model="Gustavosta/MagicPrompt-Stable-Diffusion"
        )

    @torch.inference_mode()
    def generate_prompt(self, topics: List[str]):
        prefix_prompts = [f"a {topic} image of" for topic in topics]
        prompts = prefix_prompts
        generated_prompts = self.prompt_generation_pipe(prompts)
        generated_prompts = [
            prompt[0]["generated_text"] for prompt in generated_prompts
        ]
        return generated_prompts
