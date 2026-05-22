from datasets import load_dataset
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
import json
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt


dataset = load_dataset("schneiderkamplab/danish-tool-calling-benchmark", split="danish_v1")
# dataset = load_dataset('json', data_files="data/data_processed/multi.jsonl")

# print(dataset)
# print(dataset[0]['messages'][1]['tool_calls'][0]['name'])

# print(dataset[0]['messages'][1]['tool_calls'][0]['arguments'])

# for i in range(10):
#     print(dataset[i]['messages'][1]['tool_calls'][0]['name'])

#     for key, value in dataset[i]['messages'][1]['tool_calls'][0]['arguments'].items():
#         if value != None:
#             print(f"{key} {value}")




model = AutoModelForCausalLM.from_pretrained('gemma-270m-tool-calling')
tokenizer = AutoTokenizer.from_pretrained('gemma-270m-tool-calling')

# model = AutoModelForCausalLM.from_pretrained('google/gemma-3-270m-it')
# tokenizer = AutoTokenizer.from_pretrained('google/gemma-3-270m-it')

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

def get_toolcall_and_parameters(text: str) -> list[tuple]:
    """Get the toolcall and parameters from the generated text"""
    res = []

    while text.find("<tool_call>") != -1:
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

        res.append((tool,parameters))

        text = text[index_end_toolcall + len("</tool_call>"):]

    # print(f"Toolcall: {tool}")
    # print(f"Parameters: {parameters}")

    return res




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

#Toolcall
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

#Parameters
total_parameters = {
    "location": 0,
    "unit": 0,
    "text": 0,
    "prompt": 0,
    "style": 0,
    "voice": 0,
    "query": 0
}

num_correct_parameters = {
    "location": 0,
    "unit": 0,
    "text": 0,
    "prompt": 0,
    "style": 0,
    "voice": 0,
    "query": 0
}



for i in tqdm(dataset, total=len(dataset)):
    query = format_input(i['messages'][0]['content'])
    text = pipeline(query)[0]['generated_text']

    # print(text)

    toolcalls = get_toolcall_and_parameters(text)


    # count them toolcalls and parameters from the dataset

    if i['messages'][1]['tool_calls'] != None:
        for toolcall in i['messages'][1]['tool_calls']:
            total_toolcalls[toolcall['name']] += 1

        para_meter = {k: v for k, v in i['messages'][1]['tool_calls'][0]['arguments'].items() if v is not None}

        for param in para_meter.keys():
            total_parameters[param] += 1

    # do other shit now


    for j in toolcalls:
        tool = j[0].encode('raw_unicode_escape').decode('unicode_escape')
        parameters = j[1].encode('raw_unicode_escape').decode('unicode_escape')


        try:
            if i['messages'][1]['tool_calls'] != None:

                dataset_toolcall = i['messages'][1]['tool_calls'][0]['name']

                #total_toolcalls[dataset_toolcall] += 1

                if tool == dataset_toolcall:
                    correct_toolcall += 1
                    num_correct_toolcalls[dataset_toolcall] += 1
                else:
                    print(f"Error, got: {tool}, expected: {dataset_toolcall}\n")

                para_meter = {k: v for k, v in i['messages'][1]['tool_calls'][0]['arguments'].items() if v is not None}

                #for param in para_meter.keys():
                #    total_parameters[param] += 1

                if eval(parameters).keys() == para_meter.keys():
                    for param in eval(parameters).keys():
                        num_correct_parameters[param] += 1

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

# accuracy = correct_toolcall / len(dataset)
accuracy = sum(num_correct_toolcalls.values()) / sum(total_toolcalls.values())
print(f"Tool call accuracy: {accuracy * 100}%")
# accuracy = correct_parameters / len(dataset)
accuracy = sum(num_correct_parameters.values()) / sum(total_parameters.values())
print(f"Parameter accuracy: {accuracy * 100}%")


plt.subplot(1,2,1)

plt.bar(num_correct_toolcalls.keys(), list(total_toolcalls.values()), color='r', edgecolor='black')
plt.bar(num_correct_toolcalls.keys(), list(num_correct_toolcalls.values()), color='g', edgecolor='black')

plt.ylabel('num toolcalls')
plt.title('toolcalls')

ax = plt.gca()
ax.set_ylim(bottom=0)

plt.subplot(1,2,2)

plt.bar(num_correct_parameters.keys(), list(total_parameters.values()), color='r', edgecolor='black')
plt.bar(num_correct_parameters.keys(), list(num_correct_parameters.values()), color='g', edgecolor='black')

plt.ylabel('num parameters')
plt.title('parameters')

ax = plt.gca()
ax.set_ylim(bottom=0)

plt.show()
