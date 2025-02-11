#%%
from vllm import LLM, SamplingParams
import torch

# Initialize the LLM
llm = LLM(
    model="OpenGVLab/InternVL2_5-26B-AWQ",  # Replace with your preferred multimodal model
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
    trust_remote_code=True,
    quantization="awq"
)

# Set up sampling parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=128
)

# Example prompt
prompt = "Describe what you see in this image: [image]"  # Replace [image] with actual image input

# Generate response
outputs = llm.generate([prompt], sampling_params)

# Print the generated text
for output in outputs:
    print(output.outputs[0].text)

#%%
