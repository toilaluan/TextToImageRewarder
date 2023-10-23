import importlib
import yaml
from ig_rewarding.utils import instantiate_from_config
import glob
from PIL import Image

device = "cuda"
config_file = "ig_rewarding/config/baseline.yaml"
config = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
prompter = instantiate_from_config(config["prompter"]).eval().to(device)
print("Generated Prompt:", prompter.generate_prompt(["anime"]))

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
