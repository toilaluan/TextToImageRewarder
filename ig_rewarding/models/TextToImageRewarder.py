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
        total_scores = torch.stack(
            [rewarder(images, prompt).cpu() for rewarder in self.rewarders]
        )
        sum_scores = torch.sum(total_scores * self.weights)
        total_scores = {
            rewarder_name: score.item()
            for rewarder_name, score in zip(self.rewarder_configs.keys(), total_scores)
        }
        return sum_scores.mean().item(), total_scores
