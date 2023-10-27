# %% [markdown]
# ## Synthentic Validating Text-to-image Model
# This notebook mimics the validation process when validating an T2I Endpoint
# The Validator contain 2 main components:
# 1. Prompter: Generate T2I prompt by requested topic
# 2. Rewarder: Calculate the reward of the generated images - prompt pair.

# %% [markdown]
# ### Import Library

# %%
from ig_rewarding import Validator
import pandas as pd
import json
from typing import List
import numpy as np
from transformers import Pipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ig_rewarding.utils import instantiate_from_config


# %% [markdown]
# ### Define Validating Constants

# %%
N_TRY_PER_ENDPOINT = 50
N_IMAGE_PER_PROMPT = 4

TOPICS = [
    "animated style",
    "artistic style",
    "landscape style",
    "painting style",
    "portrait style",
    "realistic photo style",
]
ENDPOINTS = {
    "sd": [
        "dreamlike-art/dreamlike-photoreal-2.0",
        "stabilityai/stable-diffusion-2-1",
        "CompVis/stable-diffusion-v1-4",
        "runwayml/stable-diffusion-v1-5",
    ],
    "sdxl": ["segmind/SSD-1B", "stabilityai/stable-diffusion-xl-base-1.0"],
}
NEGATIVE_PROMPT = "nsfw, low quality"
DEVICE = "cuda"
SAVE_DIR = "outputs"
config_file = "ig_rewarding/config/baseline.yaml"
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)


# %% [markdown]
# ### Init the Validator

# %% [markdown]
#

# %%
validator = Validator(config["rewarder"], config["prompter"], device=DEVICE)

# %% [markdown]
# ### Define Loop for Validating above Models

# %%
import shutil
import os
import json


def validate_pipe(
    validator: Validator,
    pipe: Pipeline,
    pipeline_name: str,
    topics: List[str],
    n_try: int,
    n_image_per_prompt: int,
    save_dir: str = "images",
    negative_prompt: str = "",
):
    print(f"START VALIDATE {pipeline_name}")
    raw_pipeline_name = pipeline_name
    pipeline_name = pipeline_name.replace("/", "_")
    request_result = []
    for i in range(n_try):
        request_id = i
        prompts = validator.generate_prompt(topics)

        for j in range(n_image_per_prompt):
            outputs = pipe(
                prompts,
                num_images_per_prompt=1,
                negative_prompt=[negative_prompt] * len(prompts),
            )
            file_names = [
                f"{pipeline_name}_{i}_{j*len(prompts)+t}.webp"
                for t in range(len(prompts))
            ]
            images = outputs.images
            for file_name, topic, prompt in zip(file_names, topics, prompts):
                request_result.append(
                    {
                        "file_name": file_name,
                        "topic": topic,
                        "prompt": prompt,
                        "request_id": request_id,
                        "model_type": raw_pipeline_name,
                    }
                )
            for file_name, image in zip(file_names, images):
                image.save(os.path.join(save_dir, file_name))
    return request_result


# %%
import tqdm

total_endpoints = 0
for sd_type, endpoints in ENDPOINTS.items():
    total_endpoints += len(endpoints)
progress_bar = tqdm.tqdm(total=total_endpoints)
metadata = []
shutil.rmtree(SAVE_DIR, ignore_errors=True)
os.makedirs(SAVE_DIR, exist_ok=True)
for sd_type, endpoints in ENDPOINTS.items():
    for endpoint in endpoints:
        if sd_type == "sd":
            pipe = StableDiffusionPipeline.from_pretrained(
                endpoint, torch_dtype=torch.float16
            )
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                endpoint,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        pipe = pipe.to(DEVICE)
        sub_metadata = validate_pipe(
            validator,
            pipe,
            endpoint,
            TOPICS,
            N_TRY_PER_ENDPOINT,
            N_IMAGE_PER_PROMPT,
            save_dir=SAVE_DIR,
            negative_prompt=NEGATIVE_PROMPT,
        )
        metadata.extend(sub_metadata)
        with open(os.path.join(SAVE_DIR, "metadata.jsonl"), "w") as f:
            for item in metadata:
                f.write(json.dumps(item) + "\n")
        progress_bar.update(1)
        del pipe

# %%
from datasets import load_dataset

ds = load_dataset(
    "imagefolder",
    data_dir="/root/edward/stablediffusion/ig-rewarding/outputs",
    split="train",
)

# %%

# %%
