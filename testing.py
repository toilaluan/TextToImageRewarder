from PIL import Image
from ig_rewarding.models import HPSv2

model = HPSv2(device="cuda")
image = Image.new(size=(224,224), mode="RGB")
images = [image]*4
prompt = "a full white image"

print(model(images,prompt))