from vllm import LLM, SamplingParams

import asyncio
import json
import random
import os

from openai import AsyncOpenAI

###
#
# jq -c '.[]' fil1.json > fil1.jsonl
#
###

#URL for lokal server vi vil lave inference fra.
CLIENT = [
    "http://0.0.0.0:8000/v1/"
]
client = AsyncOpenAI(base_url=CLIENT[0], api_key="")
#MODEL = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
MODEL = "google/gemma-3-270m-it"


def load_seed_data(file: str):
    output = []
    with open(file, "r", encoding="utf-8") as zike:
        for line in zike:
            output.append(json.loads(line))
        #output = zike.readlines()
        #print(output)
    return output


def save_to_file(filepath: str, response: str):
    """Write to file at filepath"""
    with open(filepath, 'a', encoding="utf-8") as f:
        f.write(response + "\n")


async def generate_text(query: str, client, model: str = MODEL) -> str:
    """Generate the text owowow"""
    try:
        chat_completion = await client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{query}\n"
                        "Generate a similar example with a variety of different prompts and styles. Only generate the jsonl part\n"
                    )
                }
            ],
            model=model,
            temperature = 0.3
        )

        response = chat_completion.choices[0].message.content
        #response = chat_completion.choices[-1].text
        print(response)
        return response
    except Exception as error:
        print(f"Error: {error}")
        return ""

async def construct_data(file: str, writefile: str):
    seeds = load_seed_data(file)

    for i in seeds:
        #print(i)
        output = await generate_text(i, client)
        if writefile != "none":
            save_to_file(writefile, output)

async def main():

    # seeds_weather = load_seed_data("data/weather_seeds.jsonl")
    # seeds_gramma  = load_seed_data("data/gramma_seeds.jsonl")
    # seeds_image   = load_seed_data("data/image_seeds.jsonl")
    # seeds_speech  = load_seed_data("data/speech_seeds.jsonl")
    # seeds_web     = load_seed_data("data/web_seeds.jsonl")

    # seeds = [seeds_weather, seeds_gramma, seeds_image, seeds_speech, seeds_web]

    # CHAIN_LENGTH = 10
    # for data in seeds:
    #     for i in range(CHAIN_LENGTH):
    #         if i == 1:
    #             seed = load_seed_data("val.json")
    #             out = asyncio.run(generate_text(data, client))
    #             save_to_file("data.json", out)
    #         else:            
    #             out = asyncio.run(generate_text(out, client))
    #             save_to_file("data.json", out)
    await asyncio.gather(construct_data("data/weather_seeds.jsonl", "data/weather_data.jsonl"),
                         construct_data("data/gramma_seeds.jsonl", "data/gramma_data.jsonl"),
                         construct_data("data/image_seeds.jsonl", "data/image_data.jsonl"),
                         construct_data("data/speech_seeds.jsonl", "data/speech_data.jsonl"),
                         construct_data("data/web_seeds.jsonl", "data/web_data.jsonl"))
    

def trim_jsonl():
    with open("data/weather_data.jsonl", "r", encoding="utf-8") as file:
        output = []
        for line in file:
            output.append(json.loads(line))

    with open("data/weather_data.jsonl", "w", encoding="utf-8") as file:
        for line in output:
            if line[0] != '{':
                pass
            else:
                file.write(line)

    # construct_data("data/weather_seeds.jsonl", "none")
    # construct_data("data/weather_seeds.jsonl", "none")
    # construct_data("data/weather_seeds.jsonl", "none")
    # construct_data("data/weather_seeds.jsonl", "none")
    # construct_data("data/weather_seeds.jsonl", "none")
    # seed = load_seed_data("val.json")
    # print(seed)


if __name__ == "__main__":
    #asyncio.run(main())
    trim_jsonl()


# async def process_seed(data, client):
#     out = data
#     for i in range(CHAIN_LENGTH):
#         if i == 1:
#             out = load_seed_data("val.json")
#         out = await generate_text(out, client)  # await, not asyncio.run()
#         save_to_file("data.json", out)

# async def main():
#     seeds = [seeds_weather, seeds_gramma, seeds_image, seeds_speech, seeds_web]
#     await asyncio.gather(*[process_seed(s, client) for s in seeds])

# if __name__ == "__main__":
#     asyncio.run(main())  # called ONCE, at the top level