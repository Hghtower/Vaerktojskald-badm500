from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm

def format_chat_with_tools(example):
    """
    Format messages for Gemma's chat template with tool calling support
    Uses a structured format that the model can learn
    """
    messages = example["messages"]
    
    conversation = []
    for msg in messages:
        role = msg["role"]
        content = msg.get("content", "")
        
        if role == "user":
            conversation.append(f"<start_of_turn>user\n{content}<end_of_turn>")
        
        elif role == "assistant":
            assistant_text = "<start_of_turn>model\n"
            
            # Check for tool calls
            if "tool_calls" in msg and msg["tool_calls"]:
                for tool_call in msg["tool_calls"]:
                    name = tool_call["name"]
                    args = tool_call["arguments"]
                    args_json = json.dumps(args)
                    assistant_text += f"<tool_call>{name}|{args_json}</tool_call>\n"
            else:
                assistant_text += f"{content}\n"
            
            assistant_text += "<end_of_turn>"
            conversation.append(assistant_text)
    
    # Join all turns
    text = "\n".join(conversation)
    
    return {"text": text}




correct = 0

dataset = load_dataset("schneiderkamplab/danish-tool-calling-benchmark", split="danish_v1")

model = AutoModelForCausalLM.from_pretrained('gemma-270m-tool-calling')
tokenizer = AutoTokenizer.from_pretrained('gemma-270m-tool-calling')

dataset = dataset.map(
    format_chat_with_tools,
    remove_columns=dataset.column_names
)

pipeline = pipeline(
    task="text-generation",
    model = model,
    #model = "gemma-270m-tool-calling",
    tokenizer = tokenizer,
    #tokenizer = "google/gemma-3-270m-it",
    # model = "google/gemma-3-270m-it",
    device = "cuda",
    torch_dtype = torch.bfloat16
)

#print(dataset[0])

text = pipeline("<start_of_turn>user\nHvad er temperaturen i Aarhus i morgen?<end_of_turn>\n")
i = text[0]['generated_text']
i = i.split('model')
print(i)


#index = i.find('<start_of_turn>model')
#print(index)

#substring = text[0]['generated_text'][i:] 
#print(substring)
# print(text)

# print("User data: " + dataset[0]['messages'][0]['content'])
# print("Gemma out: " + text[0]['generated_text'])



# for i in tqdm(dataset, total=len(dataset)):
#     text = pipeline(i['messages'][0]['content'])
#     if text == i['messages'][1]['tool_calls']:
#         correct += 1
#     print(i['messages'][0]['content'])
#     #print(i['messages'][1]['tool_calls'])
#     print(text[0]['generated_text'])

# accuracy = correct / len(dataset)
# print(f"accuracy: {accuracy}%")
