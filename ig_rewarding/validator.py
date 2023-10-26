import torch.nn as nn
from ig_rewarding.utils import instantiate_from_config
from typing import Dict, List
from PIL import Image


class Validator(nn.Module):
    def __init__(self, rewarder_cfg, prompter_cfg, device="cuda"):
        super(Validator, self).__init__()
        self.rewarder = instantiate_from_config(rewarder_cfg).to(device)
        self.prompter = instantiate_from_config(prompter_cfg).to(device)

    def generate_prompt(self, topics: List[str]):
        return self.prompter.generate_prompt(topics)

    def get_reward_score(self, images: List[Image.Image], prompt: str):
        return self.rewarder(images, prompt)
