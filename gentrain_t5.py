import torch
from transformers import pipeline

pipeline = pipeline(
    task="text2text-generation",
    model="guillaumephd/t5-french-base",
    torch_dtype=torch.float16,
    device=0
)

a = pipeline("Salut, est-ce que ça va ?")

