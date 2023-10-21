import torch.nn as nn
from utils import instantiate_from_config
from PIL import Image
from typing import List
import torch


class TextToImageRewarder(nn.Module):
    def __init__(
        self, prompt_alignment_rewarder_config, diversity_rewarder_config, alpha=0.15
    ):
        super().__init__()
        self.prompt_alignment_rewarder = instantiate_from_config(
            prompt_alignment_rewarder_config
        )
        self.diversity_rewarder = instantiate_from_config(diversity_rewarder_config)
        self.alpha = alpha

    @torch.inference_mode()
    def forward(self, images: List[Image.Image], prompt: str) -> torch.FloatTensor:
        scores = self.prompt_alignment_rewarder(images, prompt)
        diversity_rewards = self.diversity_rewarder.calculate_diversity_rewards(images)
        print(scores, diversity_rewards)
        return scores * (1 - self.alpha) + diversity_rewards * self.alpha
