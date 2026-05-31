#from vllm import LLM

import asyncio
import json
import random
import os
import multiprocessing

# from pydantic import BaseModel

from pathlib import Path
from openai import AsyncOpenAI, OpenAI
from tqdm import tqdm

###
# Used to convert json to jsonl format
# jq -c '.[]' fil1.json > fil1.jsonl
###


#URL for api host. 
CLIENT = [
    'https://api.ordbogen.ai/v1' # <-- sti ordbogens modeller (odin)
    #'http://localhost:8000/v1' # <-- sti til lokale modeller (local host)
]

client = OpenAI(base_url=CLIENT[0], api_key="")
#client = OpenAI(base_url=CLIENT[0], api_key="")

def init_worker():
    global client
    client = client

#MODEL = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit"
#MODEL = "ordbogen/gemma"
MODEL = "odin-medium"
# MODEL = "unsloth/gemma-3-27b-it-unsloth-bnb-4bit"

# How many processes to run at once when generating
NUM_PROCESSES = 5


PROMPT_GRAMMA = """You are generating training data for a large language model that learns to call tools.

Your task is to create multiple examples in the following JSON array format:

[
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
]

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `correct_grammar`.
- The "text" argument MUST EXACTLY match the sentence the user wants corrected.
  - Do NOT fix the sentence
  - Do NOT paraphrase
  - Keep all original mistakes
- The sentence should contain clear grammatical errors.
- Use different phrasings:
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

Generate 20 sampls."""

PROMPT_IMAGE = """You are generating training data for a large language model that learns to call tools.

Your task is to create multiple examples in the following JSON array format:

[
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
]

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `generate_image`.
- The "prompt" argument MUST:
  - accurately reflect what the user is asking for
  - be concise and descriptive (not a full sentence if unnecessary)
  - Use Danish for the prompt
- The "style" argument MUST be correctly inferred from the user query, and must be in english
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

Generate 20 sample."""

PROMPT_SPEECH = """You are generating training data for a large language model that learns to call tools.

Your task is to create multiple examples in the following JSON array format:

[
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
]

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
- Use many different phrasings:
  - "Kan du lave lyd af..."
  - "Læs denne tekst højt..."
  - "Kan du sige dette..."
  - "Lav en oplæsning af..."
  - "Vil du indtale følgende..."
  - "Sig dette højt..."
- Sometimes embed text inline, sometimes after punctuation
- Vary sentence length and complexity
- Use different types of content:
  - møder, beskeder, præsentationer, påmindelser, instruktioner
- Keep everything in Danish
- Output ONLY valid JSON
- Do NOT include explanations

Generate 20 samples."""

PROMPT_WEB = """You are generating training data for a large language model that learns to call tools.

Your task is to create multiple examples in the following JSON array format:

[
  {
    "messages": [
      {"role": "user", "content": "<natural user request in Danish>"},
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
]

Requirements:
- The user message MUST be natural, varied, and realistic (in Danish).
- The assistant MUST ALWAYS call the tool `search_web`.
- The "query" argument MUST correctly reflect the user's intent.
- The query should be:
  - concise
  - keyword-based (not full sentences unless necessary)
- The tool call MUST be correct and aligned with the user request.
- Use different phrasings:
  - "Find information om..."
  - "Søg efter..."
  - "Kan du finde..."
  - "Jeg leder efter..."
  - "Hvad ved man om..."
  - "Hvad er..."
  - "Find..."
  - etc.
- Include variations like:
  - "nyheder", "artikler", "forklaring", "definition", "udvikling", "spisesteder"
- Base the query on a wide variety of topics
- Include queries with both formal and informal language
- Keep everything in Danish.
- Output ONLY valid JSON.
- Do NOT include explanations.

Generate 20 samples."""

PROMPT_WEATHER = """You are generating training data for a large language model that learns to call tools." \

Your task is to create multiple examples in the following JSON array format:

[
  {
    "messages": [
      {"role": "user", "content": "<user request in Danish>"},
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
]

The new samples should be written in a variety of styles, including formal and informal language, and have the following requirements:

- The user message MUST be natural, varied, and realistic (different phrasing every time).
- The user request should contain a variety of different cities or locations, with a preference for Denmark or northern Europe.
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
  - "lige nu", "i dag", "i morgen", "senere", "om X dage", etc.
- Keep outputs strictly valid JSON.
- Do NOT include explanations.

Generate 20 sample."""

PROMPT_NONE = """
You are generating a high-quality dataset for training a tool-calling AI model.

Your task is to create multiple examples in the following JSON array format:

[
  {
    "messages": [
      {"role": "user", "content": "<user request in Danish>"},
      {
        "role": "assistant",
        "content": "<natural language response>",
        "tool_calls": []
      }
    ]
  }
]

Requirements:
- Output a valid JSON array.
- Each object must be separated by a comma.
- Do NOT include a trailing comma after the last element.
- Each example must contain a realistic Danish user query.
- The assistant MUST respond with a normal natural language answer (no tool calls).
- The "tool_calls" field must always be an empty array: [].

- Include a wide variety of queries that do NOT require tools, such as:
  - General knowledge ("Hvad er kvantefysik?")
  - Explanations ("Forklar hvad AI er")
  - Conversations ("Hvordan har du det?")
  - Advice ("Hvordan lærer jeg at programmere?")
  - Simple tasks ("Skriv en kort historie")
  - Opinions ("Hvad er den bedste film?")

- Vary Danish phrasing:
  - Questions, requests, casual conversation
  - Formal and informal tone
  - Include occasional typos or slang

- Responses should be:
  - Helpful, clear, and relevant
  - Written in Danish
  - Concise but informative

- Do NOT include any tool calls in these examples.
- Do NOT include explanations or text outside the JSON array.

- Generate 20 examples."""

PROMPT_MULTI = """
You are generating a high-quality dataset for training a tool-calling AI model.

Your task is to create multiple examples in the following JSON array format:

[
  {
    "messages": [
      {"role": "user", "content": "<user request in Danish>"},
      {
        "role": "assistant",
        "content": "",
        "tool_calls": [
          {
            "name": "<tool_name_1>",
            "arguments": { <arguments_1> }
          },
          {
            "name": "<tool_name_2>",
            "arguments": { <arguments_2> }
          }
        ]
      }
    ]
  }
]

Instructions:
- Output a valid JSON array.
- Each object must be separated by a comma.
- Do NOT include a trailing comma after the last element.

- Each example must contain a realistic Danish user query that requires MULTIPLE tool calls.
- The assistant MUST respond only with tool calls (no natural language text).
- Include at least 2 tool calls per example.

- Use a mix of available tools, such as:
  - "get_weather" → { "location": "<city in Danish>", "unit": "celsius/fahrenheit" } 
  - "search_web" → { "query": "<search query in English>" }
  - "speech_synthesis" → { "text": "<text>", "voice": "neutral/female/male" }
  - "generate_image" → { "prompt": "<image description>", "style": "<style>" }
  - "correct_grammar" → { "text": "<sentence to correct>" }

- Ensure each tool call is necessary and reflects part of the user's request.

Examples of combined intents:
- Weather + TTS:
  "Hvad er vejret i København, og kan du læse det højt?"
- Search + TTS:
  "Find info om AI og læs det op"
- Search + Weather:
  "Find info om klimaændringer og vejret i Paris"
- All three tools:
  "Find nyheder om AI, tjek vejret i Berlin og læs det hele op"

- Vary Danish phrasing:
  - Formal and informal
  - Multi-part questions
  - Use connectors like "og", "derefter", "samt"

- Rules per tool:
  - Weather:
    - "location" must be in Danish
    - Default to "celsius" unless Fahrenheit is explicitly requested
  - Search:
    - Query should usually be in English and concise
  - Speech synthesis:
    - Extract only the relevant text to be spoken
    - Default voice = "neutral" unless specified
  - Gramma correction:
    - do not paraphrase
    - do not correct spelling errors
  - Image generation:
    - "prompt" should be a concise description of the desired image
    - "style" should be inferred from the user query
- Ensure:
  - No redundant tool calls
  - Correct argument extraction
  - Logical ordering of tool calls when relevant

- Do NOT include explanations or any text outside the JSON array.
- Generate 20 examples. 
"""




prompts = [PROMPT_WEATHER, PROMPT_GRAMMA, PROMPT_IMAGE, PROMPT_SPEECH, PROMPT_WEB, PROMPT_NONE, PROMPT_MULTI]


def load_seed_data(file: str) -> list:
    """Loads the seed data from a file and returns it as a line in a list"""
    output = []
    with open(file, "r", encoding="utf-8") as zike:
        for line in zike:
            output.append(json.loads(line))
        #output = zike.readlines()
        #print(output)
    return output


def save_to_file(filepath: str, response: str) -> None:
    """Write to file at filepath"""
    with open(filepath, 'a', encoding="utf-8") as f:
        f.write(response + "\n")

def generate_text(query: str) -> str:
    """Generate text using the API client"""
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
            temperature = 0.8
        )

        #print(chat_completion)
        response = chat_completion.choices[0].message.content
        #response = chat_completion.choices[-1].text
        #print(response)
        
        return response
    except Exception as error:
        print(f"Error: {error}")
        return ""

# This is the main function that processes the dataset and generates text for each entry. 
# It uses single seeds samples as context for generation
def main(dataset, outfile: str):
    """Generate text using a single data entry as a context"""
    rows_to_process = dataset
    
    if not rows_to_process:
        print("All items processed!")
        exit()

    print(
        f"Starting processing for {len(rows_to_process)} items with {NUM_PROCESSES} processes..."
    )

    with open(outfile, "a", encoding="utf-8") as file:
        with multiprocessing.Pool(
            processes=NUM_PROCESSES, initializer=init_worker
        ) as pool:
            results = pool.imap_unordered(generate_text, rows_to_process, chunksize=1)
            for result in tqdm(results, total=len(rows_to_process)):
                try:
                    if result:
                        # result = result.replace("```json", "")
                        # result = result.replace("```", "")
                        file.write(json.dumps(json.loads(result), ensure_ascii=False) + '\n')
                        #file.write(result + '\n')
                        file.flush()
                except Exception as e:
                    print(e)
                    pass



# This is the main function that processes the dataset and generates text for each entry. 
# It uses the whole seeds file as context for generation
def main2(dataset, outfile: str):
    """Generate text using several data entries as a context"""
    rows_to_process = ""
    
    for i in range(len(dataset)):
        rows_to_process += str(dataset[i]) + "\n"

    #print(rows_to_process)

    if not rows_to_process:
        print("All items processed!")
        exit()

    print(
        f"Starting processing for {len(rows_to_process)} items with {NUM_PROCESSES} processes..."
    )

    with open(outfile, "a", encoding="utf-8") as file:
      result = generate_text(rows_to_process)
      try:
        parsed = json.loads(result)
      except Exception as e:
        print(e)
        print(result)

      #file.write(json.dumps(json.loads(result), ensure_ascii=False) + '\n')
      if isinstance(parsed, list):
          for item in parsed:
              file.write(json.dumps(item, ensure_ascii=False) + '\n')
      else:
          file.write(json.dumps(parsed, ensure_ascii=False) + '\n')
      file.flush()


## Entrypoint ##
if __name__ == "__main__":

    # Load datasett #
    # dataset_weather = load_seed_data("data/weather_odin.jsonl")
    # dataset_weather = load_seed_data("data/seeds/weather_seeds.jsonl")
    # dataset_gramma = load_seed_data("data/gramma_odin.jsonl")
    # dataset_image = load_seed_data("data/seeds/image_seeds.jsonl")
    # dataset_speech = load_seed_data("data/seeds/speech_seeds.jsonl")
    dataset_web = load_seed_data("data/seeds/web_seeds.jsonl")
    # dataset_none = load_seed_data("data/seeds/none_seeds.jsonl")
    # dataset_multi = load_seed_data("data/seeds/multi_seeds.jsonl")

    # # Set output file #
    #output_weather = "data/data_raw_v3/weather_odin.jsonl"
    # output_gramma = "data/data_raw_v3/gramma_odin.jsonl"
    #output_image = "data/data_raw_v3/image_odin.jsonl"
    #output_speech = "data/data_raw_v3/speech_odin.jsonl"
    output_web = "data/data_raw_v3/web_odin.jsonl"
    #output_none = "data/data_raw_v3/none_odin.jsonl"
    # output_multi = "data/data_raw_v3/multi_odin.jsonl"


    #prompt = prompts[0]
    # print(prompt)

    #for i in range(10):
    #  main2(dataset_weather, output_weather)
    #main(dataset_weather, output_weather)

    # prompt = prompts[1]
    # print(prompt)

    # main(dataset_gramma, output_gramma)

    #prompt = prompts[2]
    #for i in range(10):
    #    main2(dataset_image, output_image)
    # print(prompt)

    # main(dataset_image, output_image)

   # prompt = prompts[3]
   # for i in range(10):
   #   main2(dataset_speech, output_speech)
    # print(prompt)

    # main(dataset_speech, output_speech)

    prompt = prompts[4]
    # #print(prompt)

    for i in range(10):
      main2(dataset_web, output_web)

    # prompt = prompts[5]
    #print(prompt)

    # for i in range(10):
    #   main2(dataset_none, output_none)
    

    # prompt = prompts[6]
    # print(prompt)

    # for i in range(10):
    #   main2(dataset_multi, output_multi)


    print("Yay finished")

