from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    dtype = torch.bfloat16
)

# text = pipeline(f"<start_of_turn>user\nÅrhus er en by. Æ og Ø og Å. æææ<end_of_turn>\n")
# i = text[0]['generated_text']
# i = i.encode('raw_unicode_escape').decode('unicode_escape')
# print(i)

# exit()

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

total_toolcalls = {
    "get_weather": 0,
    "correct_grammar": 0,
    "generate_image": 0,
    "speech_synthesis": 0,
    "search_web": 0
}

num_correct_toolcalls = {
    "get_weather": 0,
    "correct_grammar": 0,
    "generate_image": 0,
    "speech_synthesis": 0,
    "search_web": 0
}

for i in tqdm(dataset, total=len(dataset)):
    query = format_input(i['messages'][0]['content'])
    text = pipeline(query)[0]['generated_text']

    tool, parameters = get_toolcall_and_parameters(text)

    tool = tool.encode('raw_unicode_escape').decode('unicode_escape')
    parameters = parameters.encode('raw_unicode_escape').decode('unicode_escape')

    # print(i['messages'][1]['tool_calls'][0]['arguments']) 

    #data_parameters = {k: v for k, v in i['messages'][1]['tool_calls'][0]['arguments'].items() if v is not None}

    #print(data_parameters)

    try:
        if i['messages'][1]['tool_calls'] != None:
            dataset_toolcall = i['messages'][1]['tool_calls'][0]['name']

            total_toolcalls[dataset_toolcall] += 1

            if tool == dataset_toolcall:
                correct_toolcall += 1
                num_correct_toolcalls[dataset_toolcall] += 1
            else:
                print(f"Error, got: {tool}, expected: {dataset_toolcall}\n")

            para_meter = {k: v for k, v in i['messages'][1]['tool_calls'][0]['arguments'].items() if v is not None}

            if eval(parameters) == para_meter:
                correct_parameters += 1
            else:
                print(f"Error, got: {parameters}, expected: {para_meter}\n")

        else:
            if text.find("<tool_call>") == -1:
                correct_toolcall += 1
                correct_parameters += 1
        
    except Exception as e:
        print("fuck you, me no work")
        print(f"Error: {e}")
        print(i)
        print(text)


    # if tool == 
    #     correct_toolcall += 1
    # print(i['messages'][0]['content'])
    # #print(i['messages'][1]['tool_calls'])
    # print(text[0]['generated_text'])

accuracy = correct_toolcall / len(dataset)
print(f"Tool call accuracy: {accuracy}%")
accuracy = correct_parameters / len(dataset)
print(f"Parameter accuracy: {accuracy}%")
# print(num_correct_toolcalls.values())

# total_toolcalls = {
#     "get_weather": 10,
#     "correct_grammar": 10,
#     "generate_image": 12,
#     "text_to_speech": 9,
#     "search_web": 4
# }

# num_correct_toolcalls = {
#     "get_weather": 4,
#     "correct_grammar": 9,
#     "generate_image": 3,
#     "text_to_speech": 9,
#     "search_web": 3
# }


plt.bar(num_correct_toolcalls.keys(), list(total_toolcalls.values()), color='r', edgecolor='black')
plt.bar(num_correct_toolcalls.keys(), list(num_correct_toolcalls.values()), color='g', edgecolor='black')

plt.ylabel('num toolcalls')
plt.title('toolcalls')

plt.show()
