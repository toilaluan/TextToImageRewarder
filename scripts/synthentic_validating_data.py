import shutil
import os
import json
from ig_rewarding import Validator
import pandas as pd
import json
from typing import List
from tqdm import tqdm
import numpy as np
from transformers import Pipeline
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from ig_rewarding.utils import instantiate_from_config
from argparse import ArgumentParser
synthentic_config_file = "scripts/synthentic_config.yaml"
config_file = "ig_rewarding/config/baseline.yaml"
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
synthentic_config = yaml.load(open(synthentic_config_file, "r"), Loader=yaml.FullLoader)

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
            images = []
            for prompt in prompts:
                outputs = pipe(
                    [prompt],
                    num_images_per_prompt=1,
                    negative_prompt=[negative_prompt],
                )
                images.append(outputs.images[0])
            file_names = [
                f"{pipeline_name}_{i}_{j*len(prompts)+t}.webp"
                for t in range(len(prompts))
            ]
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

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--hf_data_repo", type=str)
    parser.add_argument("--hf_token", type=str)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    validator = Validator(prompter_cfg=config["prompter"], device=DEVICE)
    total_endpoints = 0
    for sd_type, endpoints in synthentic_config.ENDPOINTS.items():
        total_endpoints += len(endpoints)
    progress_bar = tqdm.tqdm(total=total_endpoints)
    metadata = []
    with open(os.path.join(synthentic_config.SAVE_DIR, "metadata.jsonl")) as f:
        for line in f:
            metadata.append(json.loads(line))
    shutil.rmtree(synthentic_config.SAVE_DIR, ignore_errors=True)
    os.makedirs(synthentic_config.SAVE_DIR, exist_ok=True)
    for sd_type, endpoints in synthentic_config.ENDPOINTS.items():
        for endpoint in endpoints:
            if sd_type == "sd":
                pipe = StableDiffusionPipeline.from_pretrained(
                    endpoint, torch_dtype=torch.float16, safety_checker=None, requires_safety_checker=False
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    endpoint,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False
                )
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe = pipe.to(synthentic_config.DEVICE)
            sub_metadata = validate_pipe(
                validator,
                pipe,
                endpoint,
                synthentic_config.TOPICS,
                synthentic_config.N_TRY_PER_ENDPOINT,
                synthentic_config.N_IMAGE_PER_PROMPT,
                save_dir=synthentic_config.SAVE_DIR,
                negative_prompt=synthentic_config.NEGATIVE_PROMPT,
            )
            metadata.extend(sub_metadata)
            with open(os.path.join(synthentic_config.SAVE_DIR, "metadata.jsonl"), "w") as f:
                for item in metadata:
                    f.write(json.dumps(item) + "\n")
            progress_bar.update(1)
            del pipe

    from datasets import load_dataset

    ds = load_dataset(
        "imagefolder",
        data_dir="outputs",
        split="train",
    )

    ds.push_to_hub(
        args.hf_data_repo, 
        token=args.hf_token,
    )

if __name__='__main__':
    main()