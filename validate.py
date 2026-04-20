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






dataset = load_dataset("schneiderkamplab/danish-tool-calling-benchmark", split="danish_v1")

# print(dataset[0]['messages'][1]['tool_calls'][0]['name'])

# print(dataset[0]['messages'][1]['tool_calls'][0]['arguments'])

# for i in range(10):
#     print(dataset[i]['messages'][1]['tool_calls'][0]['name'])

#     for key, value in dataset[i]['messages'][1]['tool_calls'][0]['arguments'].items():
#         if value != None:
#             print(f"{key} {value}")




model = AutoModelForCausalLM.from_pretrained('gemma-270m-tool-calling')
tokenizer = AutoTokenizer.from_pretrained('gemma-270m-tool-calling')

# dataset = dataset.map(
#     format_chat_with_tools,
#     remove_columns=dataset.column_names
# )

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


# text = pipeline("<start_of_turn>user\nHvad er temperaturen i Aarhus i morgen?<end_of_turn>\n")

# i = text[0]['generated_text']


def get_toolcall_and_parameters(text: str) -> tuple:
    """Get the toolcall and parameters from the generated text"""
    
    index_toolcall = text.find("<tool_call>")
    # print(f"index of toolcall: {index_toolcall}")
    text = text[index_toolcall:]
    # print(i)
    text = text[len("<tool_call>"):]
    # print(i)

    index_split = text.find('|')
    index_end_toolcall = text.find("</tool_call>")

    tool = text[0:index_split]
    parameters = text[(index_split+1):index_end_toolcall]

    # print(f"Toolcall: {tool}")
    # print(f"Parameters: {parameters}")

    return (tool,parameters)




#index = i.find('<start_of_turn>model')
#print(index)

#substring = text[0]['generated_text'][i:]
#print(substring)
# print(text)

# print("User data: " + dataset[0]['messages'][0]['content'])
# print("Gemma out: " + text[0]['generated_text'])

def format_input(text: str) -> str:
    """Format input for model to not be bad!!! :D"""
    text = "<start_of_turn>user\n" + text + "<end_of_turn>\n"
    return text

correct_toolcall = 0
correct_parameters = 0

for i in tqdm(dataset, total=len(dataset)):
    query = format_input(i['messages'][0]['content'])
    text = pipeline(query)[0]['generated_text']

    tool, parameters = get_toolcall_and_parameters(text)

    # print(i['messages'][1]['tool_calls'][0]['arguments']) 

    #data_parameters = {k: v for k, v in i['messages'][1]['tool_calls'][0]['arguments'].items() if v is not None}

    #print(data_parameters)

    try:
        if tool == i['messages'][1]['tool_calls'][0]['name']:
            correct_toolcall += 1
        if eval(parameters) == {k: v for k, v in i['messages'][1]['tool_calls'][0]['arguments'].items() if v is not None}:
            correct_parameters += 1
        
    except:
        print("fuck you, me no work")
    # print(dataset[0]['messages'][1]['tool_calls'][0]['name'])

    # print(dataset[0]['messages'][1]['tool_calls'][0]['arguments'])



    # if tool == 
    #     correct_toolcall += 1
    # print(i['messages'][0]['content'])
    # #print(i['messages'][1]['tool_calls'])
    # print(text[0]['generated_text'])

accuracy = correct_toolcall / len(dataset)
print(f"Tool call accuracy: {accuracy}%")
accuracy = correct_parameters / len(dataset)
print(f"Parameter accuracy: {accuracy}%")
