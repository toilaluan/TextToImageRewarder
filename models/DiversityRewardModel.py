import timm
import torch.nn as nn
from PIL import Image
from typing import List
import torch
import torch.nn.functional as F


class DiversityRewardModel(nn.Module):
    def __init__(self, model: str, pretrained: bool = True):
        super().__init__()
        self.feature_extractor_model = timm.create_model(
            model, pretrained=pretrained, num_classes=0
        )
        self._init_transforms()

    def _init_transforms(self):
        data_config = timm.data.resolve_data_config(
            {}, model=self.feature_extractor_model
        )
        self.transforms = timm.data.create_transform(**data_config, is_training=False)

    def forward(self, images: torch.Tensor):
        features = self.feature_extractor_model(images)
        return features

    def _cosine_similarity_matrix(self, x: torch.Tensor, y: torch.Tensor):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return x @ y.t()

    def _min_max_normalize(self, x: torch.Tensor):
        min_value = x.min()
        max_value = x.max()
        if min_value == max_value:
            return torch.tensor([0.5] * len(x))
        return (x - min_value) / (max_value - min_value)

    def calculate_diversity_rewards(
        self, images: List[Image.Image]
    ) -> torch.FloatTensor:
        """
        Calculate diversity rewards for a batch of images.
        Return a tensor of the mean diversity reward for each image.
        """
        images = [self.transforms(image) for image in images]
        images = torch.stack(images)
        features = self.feature_extractor_model(images)
        similarity_matrix = self._cosine_similarity_matrix(features, features)
        diversity_matrix = 1 - similarity_matrix
        row_wise_diversity = [
            torch.sum(row) / (len(row) - 1) for row in diversity_matrix
        ]
        row_wise_diversity = torch.stack(row_wise_diversity)
        row_wise_diversity = self._min_max_normalize(row_wise_diversity)
        return row_wise_diversity


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
