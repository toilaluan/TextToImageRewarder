# Text To Image Rewarding
Rating Prompt - Images generated pair using mixture of expert:
1. [`Image Reward`](https://github.com/THUDM/ImageReward) `(-inf,+inf)`
2. Diversity Reward using [`timm`](https://github.com/huggingface/pytorch-image-models) pretrained models. `(0, 1)`
3. [`CLIP Interrogator`](https://github.com/pharmapsychotic/clip-interrogator) + [`Bert Score`](https://github.com/Tiiiger/bert_score) reward `(0, 1)`

Final reward is a weighted sum of all the above rewards. Change this in the config file `ig_rewarding/config/baseline.yaml`

## Getting Started
### Installation
```bash
pip install -r requirements.txt
```

### Example
```python
import importlib
import yaml
from ig_rewarding.utils import instantiate_from_config
import glob
from PIL import Image

device = "cuda"
config_file = "ig_rewarding/config/baseline.yaml"
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

model = instantiate_from_config(config["rewarder"]).eval().to(device)
prompt = (
    "a painting of an ocean with clouds and birds, day time, low depth field effect"
)
image_folder = "assets/images"
images = glob.glob(f"{image_folder}/*")
images = [Image.open(image) for image in images]
print("Model Score", model(images, prompt))

image_folder = "assets/duplicated_images"
images = glob.glob(f"{image_folder}/*")
images = [Image.open(image) for image in images]
print("Model Score:", model(images, prompt))
```
Output is something like that
```
prompt_alignment_rewarder: -0.6467663645744324
diversity_rewarder: 0.6754872798919678
clip_interrogator_rewarder: 0.18380855023860931
Model Score tensor(-0.5179)

prompt_alignment_rewarder: 0.5811619758605957
diversity_rewarder: -9.21034049987793
clip_interrogator_rewarder: 0.19186104834079742
Model Score: tensor(-0.7716)
```