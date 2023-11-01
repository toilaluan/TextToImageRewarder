import torch.nn as nn
from ig_rewarding.utils import instantiate_from_config
from typing import Dict, List
from PIL import Image


class Validator(nn.Module):
    def __init__(self, rewarder_cfg=None, prompter_cfg=None, device="cuda"):
        super(Validator, self).__init__()
        if rewarder_cfg:
            self.rewarder = instantiate_from_config(rewarder_cfg).to(device)
        if prompter_cfg:
            self.prompter = instantiate_from_config(prompter_cfg).to(device)

    def generate_prompt(self, topics: List[str], n_prompts: int = 100):
        return self.prompter.generate_prompt(topics, n_prompts)

    def get_reward_score(self, images: List[Image.Image], prompt: str):
        return self.rewarder(images, prompt)
