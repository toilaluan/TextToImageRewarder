import torch.nn as nn
from ig_rewarding.utils import instantiate_from_config
from PIL import Image
from typing import List
import torch


class TextToImageRewarder(nn.Module):
    def __init__(self, rewarder_configs: dict):
        super().__init__()
        self.rewarders = nn.ModuleList(
            [instantiate_from_config(config) for config in rewarder_configs.values()]
        )
        self.weights = torch.tensor(
            [config["weight"] for config in rewarder_configs.values()]
        )
        self.rewarder_configs = rewarder_configs

    @torch.inference_mode()
    def forward(self, images: List[Image.Image], prompt: str) -> torch.FloatTensor:
        scores = torch.stack([rewarder(images, prompt).cpu() for rewarder in self.rewarders])
        for rewarder_name, score in zip(self.rewarder_configs.keys(), scores):
            print(f"{rewarder_name}: {score}")
        scores = torch.sum(scores * self.weights)
        return scores.mean()
