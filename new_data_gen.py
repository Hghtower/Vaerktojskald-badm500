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

#sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def load_llm():
    # Create an LLM.
    llm = LLM(model="unsloth/gemma-3-4b-it-unsloth-bnb-4bit", model_impl="transformers")
    #llm = LLM(model="unsloth/gemma-3-4b-it-bnb-4bit", model_impl="transformers")
    #llm = LLM(model="google/gemma-3-4b-it")
    #llm = LLM(model="unsloth/gemma-3-1b-it", model_impl="transformers")

    return llm

def load_seed_data():
    pass

async def _generate_text(query: str, client) -> str:
    chat_completion = await client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": (
                    f"{query}\n"
                    "Generate the appropriate in danish\n"
                )
            }
        ],
        model="unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        temperature = 0.3
    )

    response = chat_completion.choices[0].message.content
    print(response)
    #print("Failed to do shit")
    #return ""

    return response

def main():
    with open("train_weather.json", "r") as zike:
        output = zike.readlines()

    print(str(output))

    #asyncio.run(_generate_text("Hvordan er vejret i Paris?", client))

if __name__ == "__main__":
    main()

