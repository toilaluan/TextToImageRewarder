import timm
import torch.nn as nn
from PIL import Image
from typing import List
import torch
import torch.nn.functional as F


class DiversityRewardModel(nn.Module):
    def __init__(self, model: str, pretrained: bool = True, device: str = "cuda"):
        super().__init__()
        self.feature_extractor_model = timm.create_model(
            model, pretrained=pretrained, num_classes=0
        ).to(device)
        self.device = device
        self._init_transforms()

    def _init_transforms(self):
        data_config = timm.data.resolve_data_config(
            {}, model=self.feature_extractor_model
        )
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def _cosine_similarity_matrix(self, x: torch.Tensor, y: torch.Tensor):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return x @ y.t()

    def extract_feature(self, x):
        features = self.feature_extractor_model.forward_features(x)
        if len(features.shape) == 4:
            N, C, H, W = features.shape
            features = features.view(N, C, -1).permute(0, 2, 1).contiguous()
        features = features.view(features.shape[0], -1)
        return features

    def forward(self, images: List[Image.Image], *args, **kwargs) -> torch.FloatTensor:
        """
        Calculate diversity rewards for a batch of images.
        Return a tensor of the mean diversity reward for each image.
        """
        images = [self.transforms(image) for image in images]
        images = torch.stack(images).to(self.device)
        features = self.extract_feature(images)
        similarity_matrix = self._cosine_similarity_matrix(features, features)
        diversity_matrix = 1 - similarity_matrix
        diversity_rewards = diversity_matrix.sum(dim=1) / (
            diversity_matrix.shape[0] - 1
        )
        diversity_rewards = diversity_rewards.mean()
        if diversity_rewards < 1e-1:
            return (diversity_rewards + 1e-4).log()
        return diversity_rewards


if __name__ == "__main__":
    model = DiversityRewardModel("vgg16_bn")
    print(model)
    print(
        model.calculate_diversity_rewards(
            [
                Image.new(size=(224, 224), mode="RGB", color="blue"),
                Image.new(size=(224, 224), mode="RGB", color="red"),
            ]
        )
    )
