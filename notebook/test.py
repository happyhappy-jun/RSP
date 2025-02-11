#%%
import torch
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def load_image(image_file, input_size=448):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    pixel_values = transform(image).unsqueeze(0)
    return pixel_values

# Initialize model and tokenizer
model_path = "OpenGVLab/InternVL2_5-26B"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True
).eval().cuda()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

# Load and process image
image_path = "examples/image1.jpg"  # Replace with your image path
pixel_values = load_image(image_path).to(torch.bfloat16).cuda()

# Generate text
generation_config = dict(max_new_tokens=1024, do_sample=True)

# Single-image conversation
question = '<image>\nPlease describe this image in detail.'
response = model.chat(tokenizer, pixel_values, question, generation_config)
print(f'User: {question}\nAssistant: {response}')

# Multi-turn conversation
question = '<image>\nWhat do you see in this image?'
response, history = model.chat(tokenizer, pixel_values, question, generation_config, history=None, return_history=True)
print(f'User: {question}\nAssistant: {response}')

follow_up = 'What emotions or mood does this image convey?'
response, history = model.chat(tokenizer, pixel_values, follow_up, generation_config, history=history, return_history=True)
print(f'User: {follow_up}\nAssistant: {response}')

#%%
