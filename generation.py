from vllm import LLM, SamplingParams

import asyncio
import json
import random
import os

from openai import AsyncOpenAI

#URL for lokal server vi vil lave inference fra.
CLIENT = [
    "http://0.0.0.0:8000/v1/"
]
client = AsyncOpenAI(base_url=CLIENT[0], api_key="")
MODEL = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"

def load_seed_data(file: str):
    with open(file, "r") as zike:
        output = json.load(zike)
        #output = zike.readlines()
        print(output)
    return str(output)


def save_to_file(filepath: str, response: str):
    """Write to file at filepath"""
    with open(filepath, 'a') as f:
        f.write(response + "\n")


async def generate_text(query: str, client, model: str = MODEL) -> str:
    """Generate the text owowow"""
    try:
        chat_completion = await client.completions.create(
            prompt=[
                {
                    "role": "user",
                    "content": (
                        f"{query}\n"
                        "Generate a similar example with a variety of different prompts and styles. Only generate the json part\n"
                    )
                }
            ],
            #model="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
            model=model,
            n = 10,
            #temperature = 0.3
        )

        #response = chat_completion.choices[0].message.content
        response = chat_completion.choices[-1].text
        print(response)
        return response
    except Exception as error:
        print(f"Error: {error}")
        return ""

async def construct_data(file: str, writefile: str):
    seed = load_seed_data(file)

    for i in seed:
        output = asyncio.run(generate_text(i, client))
        save_to_file(writefile, output)

async def main():

    # seeds_weather = load_seed_data("weatherfil")
    # seeds_gramma  = load_seed_data("grammafil")
    # seeds_image   = load_seed_data("imagefil")
    # seeds_speech  = load_seed_data("speech")
    # seeds_web     = load_seed_data("webfil")

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
    await asyncio.gather(construct_data("Weather"),
                         construct_data("image"),
                         construct_data("gramma"),
                         construct_data("speech"),
                         construct_data("web"))
    # seed = load_seed_data("val.json")
    # print(seed)


if __name__ == "__main__":
    asyncio.run(main())



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