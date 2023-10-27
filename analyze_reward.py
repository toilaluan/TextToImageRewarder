# %%
import json
from datasets import load_dataset, Image
from ig_rewarding.utils import instantiate_from_config
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ig_rewarding import Validator


# %%
config_file = "ig_rewarding/config/baseline.yaml"
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

dataset_name = "toilaluan/t2i_topic_comparision_db_v2"
save_file = f"{dataset_name.replace('/', '-')}.json"

# %%
dataset = load_dataset(dataset_name, split="train")
dataset = dataset.cast_column("image", Image(decode=True))
dataset[0]["image"]

# %%
dataset_df = dataset.to_pandas()
dataset_df.head()

# %%
# group by model_type
dataset_df.groupby("model_type").count()

# %%
validator = Validator(config["rewarder"], config["prompter"], device="cuda")

# %%
import io
from tqdm import tqdm
from PIL import Image

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
            sum_reward, individual_rewards = validator.get_reward_score(images, prompt)

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

# %%
