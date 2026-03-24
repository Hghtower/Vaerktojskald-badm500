from vllm import LLM

import asyncio
import json
import random
import os
import multiprocessing


from pathlib import Path
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm
###
#
# jq -c '.[]' fil1.json > fil1.jsonl
#
###

#URL for lokal server vi vil lave inference fra.
CLIENT = [
    #'https://api.ordbogen.ai/v1'
    'http://localhost:8000/v1'
]
#client = OpenAI(base_url=CLIENT[0], api_key="vopdhQmXsNzx7bKEt0qlUKzKDef8Q9wioKHVW7snc3a52584")
client = OpenAI(base_url=CLIENT[0], api_key="")
def init_worker():
    global client
    client = client

#MODEL = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
#MODEL = "ordbogen/gemma"
#MODEL = "odin-medium"
MODEL = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"
NUM_PROCESSES = 40

PROMPT_GRAMMA = """You are generating training data for a large language model that learns to call tools.

Your task:
Create multiple JSON samples. Each sample must follow this exact structure:

{
  "messages": [
    {"role": "user", "content": "<natural user request in Danish>"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "name": "correct_grammar",
          "arguments": {
            "text": "<sentence to correct>"
          }
        }
      ]
    }
  ]
}

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `correct_grammar`.
- The "text" argument MUST EXACTLY match the sentence the user wants corrected.
  - Do NOT fix the sentence
  - Do NOT paraphrase
  - Keep all original mistakes
- The sentence should contain clear grammatical errors.

Variation guidelines:
- Use many different phrasings:
  - "Kan du rette denne sætning..."
  - "Tjek grammatikken i..."
  - "Lyder dette korrekt..."
  - "Kan du hjælpe med denne..."
  - "Jeg tror der er fejl i..."
- Sometimes include punctuation before the sentence, sometimes not
- Sometimes embed the sentence mid-text, sometimes at the end
- Vary error types:
  - verb tense errors
  - word order mistakes
  - incorrect pronouns
  - agreement errors
- Vary sentence length and complexity

Rules:
- Output ONLY valid JSON
- Do NOT include explanations
- Assistant "content" must always be empty
- Each sample must be unique
- The tool call MUST always be correct and aligned with the user request

Generate 5 samples."""

PROMPT_IMAGE = """You are generating training data for a large language model that learns to call tools.

Your task:
Create multiple JSON samples. Each sample must follow this exact structure:

{
  "messages": [
    {"role": "user", "content": "<natural user request in Danish>"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "name": "generate_image",
          "arguments": {
            "prompt": "<image description>",
            "style": "<style>"
          }
        }
      ]
    }
  ]
}

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `generate_image`.
- The "prompt" argument MUST:
  - accurately reflect what the user is asking for
  - be concise and descriptive (not a full sentence if unnecessary)
- The "style" argument MUST be correctly inferred:
  - "realistic" for real-world scenes
  - "fantasy" for magical or mythical content
  - "cyberpunk" for futuristic neon/city themes
  - "futuristic" for sci-fi (non-cyberpunk) settings

Variation guidelines:
- Use different phrasings:
  - "Kan du lave et billede af..."
  - "Generér et billede af..."
  - "Jeg vil gerne have et billede af..."
  - "Skab et billede af..."
  - "Lav en illustration af..."
- Vary structure:
  - with/without polite phrases
  - short vs long requests
- Mix subjects:
  - natur, byer, dyr, mennesker, sci-fi, fantasy
- Vary detail level in user requests

Rules:
- Output ONLY valid JSON
- Do NOT include explanations
- Assistant "content" must always be empty
- Each sample must be unique
- The tool call MUST always be correct and aligned with the request

Generate 5 samples."""

PROMPT_SPEECH = """You are generating training data for a large language model that learns to call tools.

Your task:
Create multiple JSON samples. Each sample must follow this exact structure:

{
  "messages": [
    {"role": "user", "content": "<natural user request in Danish>"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "name": "text_to_speech",
          "arguments": {
            "text": "<exact text to read>",
            "voice": "<voice type>"
          }
        }
      ]
    }
  ]
}

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `text_to_speech`.
- The "text" argument MUST EXACTLY match the text the user wants read aloud.
  - Do NOT paraphrase
  - Do NOT modify punctuation
- The "voice" argument must be:
  - "female" if the user explicitly asks for a female voice
  - "male" if the user explicitly asks for a male voice
  - "neutral" if no voice is specified

Variation guidelines:
- Use many different phrasings:
  - "Kan du lave lyd af..."
  - "Læs denne tekst højt..."
  - "Kan du sige dette..."
  - "Lav en oplæsning af..."
  - "Vil du indtale følgende..."
- Sometimes include colon, sometimes not
- Sometimes embed text inline, sometimes after punctuation
- Vary sentence length and complexity
- Use different types of content:
  - møder, beskeder, præsentationer, påmindelser, instruktioner
- Keep everything in Danish

Rules:
- Output ONLY valid JSON
- Do NOT include explanations
- Assistant "content" must always be empty
- Each sample must be unique
- The tool call must always be correct and aligned with the user request

Generate 5 samples."""

PROMPT_WEB = """You are generating training data for a large language model that learns to call tools.

Your task:
Create multiple JSON samples. Each sample must follow this exact structure:

{
  "messages": [
    {"role": "user", "content": "<natural user request>"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "name": "search_web",
          "arguments": {"query": "<search query>"}
        }
      ]
    }
  ]
}

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `search_web`.
- The "query" argument MUST correctly reflect the user's intent.
- The query should be:
  - concise
  - keyword-based (not full sentences unless necessary)
- The tool call MUST be correct and aligned with the user request.

Variation guidelines:
- Use different phrasings:
  - "Find information om..."
  - "Søg efter..."
  - "Kan du finde..."
  - "Jeg leder efter..."
  - "Hvad ved man om..."
- Mix topics:
  - teknologi, sundhed, klima, sport, økonomi, historie osv.
- Include variations like:
  - "nyheder", "artikler", "forklaring", "definition", "udvikling"
- Vary complexity:
  - short queries ("AI udvikling")
  - longer intent ("nyeste forskning i kræftbehandling")
- Keep everything in Danish.

Rules:
- Output ONLY valid JSON.
- Do NOT include explanations.
- Do NOT include assistant text (content must be empty).
- Ensure each sample is different.
- Ensure the query always matches the user request accurately.

Generate 5 samples."""

PROMPT_WEATHER = """You are generating training data for a large language model that learns to call tools." \

"Your task:"
Create multiple JSON samples. The samples must follow this exact structure:

{
  "messages": [
    {"role": "user", "content": "<natural user request>"},
    {
      "role": "assistant",
      "content": "",
      "tool_calls": [
        {
          "name": "<tool_name>",
          "arguments": {<correct arguments>}
        }
      ]
    }
  ]
}

Requirements:
- The user message MUST be natural, varied, and realistic (different phrasing every time).
- The user request should contain a variety of different cities or locations, with a preference for Denmark or northern Europe.
- The assistant MUST ALWAYS call the correct tool.
- The tool name MUST be exactly: get_weather
- The arguments MUST ALWAYS be correct and extracted from the user message:
  - "location": the city or area mentioned
  - "unit": 
      - "celsius" unless the user explicitly asks for Fahrenheit
      - "fahrenheit" if the user mentions Fahrenheit
- Do NOT include any assistant text, only tool_calls.
- Vary phrasing a lot:
  - Questions, casual phrasing, formal phrasing, short queries, long queries
- Vary intent slightly:
  - "lige nu", "i dag", "i morgen", "senere", etc.
- Keep outputs strictly valid JSON.
- Do NOT include explanations.

Generate 5 samples."""

prompts = [PROMPT_WEATHER, PROMPT_GRAMMA, PROMPT_IMAGE, PROMPT_SPEECH, PROMPT_WEB]

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

#Generate a similar example in Danish. Only generate valid json\n"
prompt = ""
def generate_text(query: str) -> str:
    """Generate the text owowow"""
    try:
        #print(ordbogen_client)
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{query}\n"
                        f"{prompt}"
                        #"Generate Similar example. USE JSON"
                    ),
                }
            ],
            model=MODEL,
            temperature = 0.7
        )

        #print(chat_completion)
        response = chat_completion.choices[0].message.content
        #response = chat_completion.choices[-1].text
        #print(response)
        return response
    except Exception as error:
        print(f"Error: {error}")
        return ""

# async def construct_data(file: str, writefile: str):
#     seeds = load_seed_data(file)
#
#     for i in seeds:
#         #print(i)
#         output = await generate_text(i, client)
#         if writefile != "none":
#             save_to_file(writefile, output)

def main(dataset, outfile: str):
    rows_to_process = dataset

    if not rows_to_process:
        print("All items processed!")
        exit()

    print(
        f"Starting processing for {len(dataset_weather)} items with {NUM_PROCESSES} processes..."
    )

    with open(outfile, "a", encoding="utf-8") as file:
        with multiprocessing.Pool(
            processes=NUM_PROCESSES, initializer=init_worker
        ) as pool:
            results = pool.imap_unordered(generate_text, rows_to_process, chunksize=1)
            for result in tqdm(results, total=len(rows_to_process)):
                try:
                    if result:
                        result = result.replace("```json", "")
                        result = result.replace("```", "")
                        file.write(json.dumps(json.loads(result)) + '\n')
                        file.flush()
                except Exception as e:
                    print(e)
                    pass

## Entrypoint ##
if __name__ == "__main__":

    # Load datasett #
    dataset_weather = load_seed_data("data/weather_seeds.jsonl")
    dataset_gramma = load_seed_data("data/gramma_seeds.jsonl")
    dataset_image = load_seed_data("data/image_seeds.jsonl")
    dataset_speech = load_seed_data("data/speech_seeds.jsonl")
    dataset_web = load_seed_data("data/web_seeds.jsonl")
    
    # Set output file #
    output_weather = "data/weather_target.jsonl"
    output_gramma = "data/gramma_target.jsonl"
    output_image = "data/image_target.jsonl"
    output_speech = "data/speech_target.jsonl"
    output_web = "data/web_target.jsonl"
    
    prompt = prompts[0]
    print(prompt)

    main(dataset_weather, output_weather)

    prompt = prompts[1]
    print(prompt)

    main(dataset_gramma, output_gramma)

    prompt = prompts[2]
    print(prompt)

    main(dataset_image, output_image)

    prompt = prompts[3]
    print(prompt)

    main(dataset_speech, output_speech)

    prompt = prompts[4]
    print(prompt)

    main(dataset_web, output_web)

    print("Yay finished")

