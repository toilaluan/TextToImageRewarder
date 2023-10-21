import importlib
import yaml
from ig_rewarding.utils import instantiate_from_config
import glob
from PIL import Image

config_file = "ig_rewarding/config/baseline.yaml"
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

model = instantiate_from_config(config["rewarder"])
prompt = (
    "a painting of an ocean with clouds and birds, day time, low depth field effect"
)
image_folder = "assets/images"
images = glob.glob(f"{image_folder}/*")
images = [Image.open(image) for image in images]
print(model(images, prompt))
