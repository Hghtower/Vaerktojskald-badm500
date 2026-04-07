from datasets import load_dataset
import torch
import torch.nn as nn
from transformers import AutoTokenizer, pipeline
import json
from tqdm import tqdm

correct = 0

dataset = load_dataset("schneiderkamplab/danish-tool-calling-benchmark", split="danish_v1")

pipeline = pipeline(
    task="text-generation",
    #model = "gemma-270m-tool-calling",
    model = "google/gemma-3-270m-it",
    device = "cpu",
    torch_dtype = torch.bfloat16
)

# text = pipeline(text_inputs=dataset[0]['messages'][0]['content'])
# print("User data: " + dataset[0]['messages'][0]['content'])
# print("Gemma out: " + text[0]['generated_text'])

for i in tqdm(dataset, total=len(dataset)):
    text = pipeline(i['messages'][0]['content'])
    if text == i['messages'][1]['tool_calls']:
        correct += 1
    #print(i['messages'][0]['content'])
    #print(i['messages'][1]['tool_calls'])
    #print(text[0]['generated_text'])

accuracy = correct / len(dataset)
print(f"accuracy: {accuracy}%")
