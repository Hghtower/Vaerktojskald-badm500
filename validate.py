from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import AutoTokenizer, pipeline
import json

dataset = load_dataset("schneiderkamplab/danish-tool-calling-benchmark", split="danish")

pipeline = pipeline(
    task="text-generation",
    #model = "gemma-270m-tool-calling",
    model = "google/gemma-3-270m-it",
    device = "mps",
    torch_dtype = torch.bfloat16
)

# text = pipeline(text_inputs=dataset[0]['messages'][0]['content'])
# print("User data: " + dataset[0]['messages'][0]['content'])
# print("Gemma out: " + text[0]['generated_text'])

for i in dataset:
    text = pipeline(i['messages'][0]['content'])
    print(i['messages'][0]['content'])
    print(text[0]['generated_text'])