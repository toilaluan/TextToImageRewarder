import shutil
import os
import json
from ig_rewarding import Validator
import pandas as pd
import json
from typing import List, Dict
import tqdm
import numpy as np
import easydict
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
synthentic_config = easydict.EasyDict(synthentic_config)


def validate_pipe(
    pipe: Pipeline,
    pipeline_name: str,
    n_try: int,
    n_image_per_prompt: int,
    all_prompts: Dict[str, List[str]],
    save_dir: str = "images",
    negative_prompt: str = "",
):
    print(f"START VALIDATE {pipeline_name}")
    raw_pipeline_name = pipeline_name
    pipeline_name = pipeline_name.replace("/", "_")
    request_result = []

    for i in range(n_try):
        request_id = i
        prompts = []
        topic_names = []
        for topic_name, prompt in all_prompts.items():
            topic_names.append(topic_name)
            prompts.append(prompt[i])
        print(prompts)
        for j, prompt in enumerate(prompts):
            outputs = pipe(
                [prompt],
                num_images_per_prompt=n_image_per_prompt,
                negative_prompt=[negative_prompt],
            )
            images = outputs.images
            file_names = [
                f"{pipeline_name}_{i}_{j*n_image_per_prompt+t}.webp"
                for t in range(n_image_per_prompt)
            ]
            for file_name in file_names:
                request_result.append(
                    {
                        "file_name": file_name,
                        "topic": topic_names[j],
                        "prompt": prompt,
                        "request_id": request_id,
                        "model_type": raw_pipeline_name,
                    }
                )
            for file_name, image in zip(file_names, images):
                image.save(os.path.join(save_dir, file_name))
        with open("temp.jsonl", "w") as f:
            for item in request_result:
                f.write(json.dumps(item) + "\n")
    return request_result


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--hf_data_repo", type=str)
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--continue_generate", action="store_true")

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    validator = Validator(
        prompter_cfg=config["prompter"], device=synthentic_config.DEVICE
    )
    total_endpoints = 0
    for sd_type, endpoints in synthentic_config.ENDPOINTS.items():
        total_endpoints += len(endpoints)
    progress_bar = tqdm.tqdm(total=total_endpoints)
    metadata = []
    infered_endpoints = []
    if args.continue_generate:
        with open(os.path.join(synthentic_config.SAVE_DIR, "metadata.jsonl")) as f:
            for line in f:
                metadata.append(json.loads(line))
        for item in metadata:
            infered_endpoints.append(item["model_type"])
        infered_endpoints = list(set(infered_endpoints))
    else:
        shutil.rmtree(synthentic_config.SAVE_DIR, ignore_errors=True)
        os.makedirs(synthentic_config.SAVE_DIR, exist_ok=True)
    print("Infered endpoints: ", infered_endpoints)
    topic_names = list(synthentic_config.TOPICS.keys())
    topic_prefixes = [
        synthentic_config.TOPICS[topic_name]["prefix"] for topic_name in topic_names
    ]
    with torch.no_grad():
        prompts = validator.generate_prompt(
            topic_prefixes, synthentic_config.N_TRY_PER_ENDPOINT
        )
    prompts = {
        topic_name: prompt_topic_set
        for topic_name, prompt_topic_set in zip(topic_names, prompts)
    }
    del validator
    for sd_type, endpoints in synthentic_config.ENDPOINTS.items():
        torch.cuda.empty_cache()
        for endpoint in endpoints:
            if endpoint in infered_endpoints:
                continue
            if sd_type == "sd":
                pipe = StableDiffusionPipeline.from_pretrained(
                    endpoint,
                    torch_dtype=torch.float16,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            else:
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    endpoint,
                    torch_dtype=torch.float16,
                    use_safetensors=True,
                    variant="fp16",
                    safety_checker=None,
                    requires_safety_checker=False,
                )
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
            pipe.to(synthentic_config.DEVICE)
            # pipe.enable_model_cpu_offload()
            sub_metadata = validate_pipe(
                pipe=pipe,
                pipeline_name=endpoint,
                n_try=synthentic_config.N_TRY_PER_ENDPOINT,
                n_image_per_prompt=synthentic_config.N_IMAGE_PER_PROMPT,
                save_dir=synthentic_config.SAVE_DIR,
                negative_prompt=synthentic_config.NEGATIVE_PROMPT,
                all_prompts=prompts,
            )
            metadata.extend(sub_metadata)
            with open(
                os.path.join(synthentic_config.SAVE_DIR, "metadata.jsonl"), "w"
            ) as f:
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


if __name__ == "__main__":
    main()
