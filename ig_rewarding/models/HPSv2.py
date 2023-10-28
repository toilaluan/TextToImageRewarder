import torch.nn as nn
import torch
from hpsv2.img_score import *
from PIL import Image
import torch
from typing import List
import numpy as np


class HPSv2(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.min = 0.2
        self.max = 0.3
        self.device = device
        self.model, self.preprocess, self.tokenizer = self._init_model()

    def forward(self, images: List[Image.Image], prompt: str, *args, **kwargs):
        # Process the image
        rewards = []
        for image in images:
            image = (
                self.preprocess(image).unsqueeze(0).to(device=device, non_blocking=True)
            )
            # Process the prompt
            text = self.tokenizer([prompt]).to(device=device, non_blocking=True)
            # Calculate the HPS
            with torch.cuda.amp.autocast():
                outputs = self.model(image, text)
                image_features, text_features = (
                    outputs["image_features"],
                    outputs["text_features"],
                )
                logits_per_image = image_features @ text_features.T

                hps_score = torch.diagonal(logits_per_image).cpu()
            rewards.append(hps_score)
        score = torch.tensor(rewards).mean() 
        return (score - self.min) / (self.max - self.min)

    def _init_model(self, cp: str = os.path.join(root_path, "HPS_v2_compressed.pt")):
        initialize_model()
        model = model_dict["model"]
        preprocess_val = model_dict["preprocess_val"]

        # check if the checkpoint exists
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        if cp == os.path.join(root_path, "HPS_v2_compressed.pt") and not os.path.exists(
            cp
        ):
            print("Downloading HPS_v2_compressed.pt ...")
            url = "https://huggingface.co/spaces/xswu/HPSv2/resolve/main/HPS_v2_compressed.pt"
            r = requests.get(url, stream=True)
            with open(os.path.join(root_path, "HPS_v2_compressed.pt"), "wb") as HPSv2:
                total_length = int(r.headers.get("content-length"))
                for chunk in progress.bar(
                    r.iter_content(chunk_size=1024),
                    expected_size=(total_length / 1024) + 1,
                ):
                    if chunk:
                        HPSv2.write(chunk)
                        HPSv2.flush()
            print(
                "Download HPS_2_compressed.pt to {} sucessfully.".format(
                    root_path + "/"
                )
            )

        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        tokenizer = get_tokenizer("ViT-H-14")
        model = model.to(self.device)
        model.eval()
        return model, preprocess_val, tokenizer
