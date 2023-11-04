import json
from datasets import load_dataset, Image
from ig_rewarding.utils import instantiate_from_config
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ig_rewarding import Validator
import argparse
import io
from tqdm import tqdm
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_hf_data_repo", type=str)
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--input_hf_data_repo", type=str)

    args = parser.parse_args()
    return args


def main():
    args = get_args()
    config_file = "ig_rewarding/config/baseline.yaml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    validator = Validator(rewarder_cfg=config["rewarder"], device="cuda")

    dataset_name = args.input_hf_data_repo
    save_file = f"{dataset_name.replace('/', '-')}.json"

    dataset = load_dataset(dataset_name, split="train", download_mode='force_redownload', cache_dir='./cache')
    dataset_df = dataset.to_pandas()

    rewards = []
    progress_bar = tqdm(
        total=len(dataset_df.groupby(["model_type", "request_id", "topic"]))
    )
    for model_type, model_group in dataset_df.groupby("model_type"):
        for request_id, request_group in model_group.groupby("request_id"):
            for topic, topic_group in request_group.groupby("topic"):
                images = topic_group["image"]
                images = [
                    Image.open(io.BytesIO(image["bytes"])).convert("RGB")
                    for image in images
                ]
                prompt = topic_group["prompt"].iloc[0]
                sum_reward, individual_rewards = validator.get_reward_score(
                    images, prompt
                )

                rewards.append(
                    {
                        "model_type": model_type,
                        "request_id": request_id,
                        "topic": topic,
                        "reward": sum_reward,
                        "individual_rewards": individual_rewards,
                    }
                )

                with open(save_file, "w") as f:
                    for item in rewards:
                        f.write(json.dumps(item) + "\n")
                progress_bar.update(1)
    progress_bar.close()

    with open(save_file, "w") as f:
        for item in rewards:
            f.write(json.dumps(item) + "\n")

    ds = load_dataset("json", data_files=save_file)
    ds.push_to_hub(args.save_hf_data_repo, token=args.hf_token)


if __name__ == "__main__":
    main()
