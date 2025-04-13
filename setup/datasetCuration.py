from datasets import load_dataset
from IPython.display import display
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

import requests
import torch

dataset = load_dataset('HuggingFaceM4/FairFace', '1.25')
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

trainingData = dataset['train']
print(trainingData[0]['image'])
img = trainingData[0]['image']
display(img)

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)

prompt = "<OD>"

url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

generated_ids = model.generate(
    input_ids=inputs["input_ids"],
    pixel_values=inputs["pixel_values"],
    max_new_tokens=1024,
    do_sample=False,
    num_beams=3,
)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))

print(parsed_answer)

torch.cuda.empty_cache()

activation = {}

def get_activation(name):
  def hook(model, input, output):
    activation[name] = output
  return hook

model.vision_tower.register_forward_hook(get_activation('DaVIT'))

for i in range(len(model.vision_tower.blocks)):
  model.vision_tower.blocks[i].register_forward_hook(get_activation('DaVIT_block_' + str(i)))

prompt = "<OD>"

latentArray = [[], [], [], []]

for i in range(0, 4000):

  image = trainingData[i]['image']

  inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

  generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      do_sample=False,
      num_beams=3,
  )

  k = 0
  for key in activation.keys():
      print(key, activation[key][0].shape)
      latentArray[k].append(activation[key][0].detach().cpu())  # <-- move to CPU
      k += 1

  activation.clear()
  torch.cuda.empty_cache()

  print(i)

for i, layer_outputs in enumerate(latentArray):
    torch.save(layer_outputs, f"latent_block_{i}.pt")

latent_block_0 = torch.load("latent_block_0.pt")
print(latent_block_0[0].shape)