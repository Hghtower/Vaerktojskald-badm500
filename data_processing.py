import pandas as pd
import json

data_weather_gemma = pd.read_json("data/data_raw/weather2.jsonl", lines=True)
data_weather_odin = pd.read_json('data/data_raw/weather_odin.jsonl', lines=True)
dat = [data_weather_odin, data_weather_gemma]
data_weather = pd.concat(dat)
data_weather = data_weather.drop_duplicates()
data_weather.to_json('data/data_processed/weather.jsonl', orient='records', lines=True, force_ascii=False)

data_gramma_odin = pd.read_json('data/data_raw/gramma_odin.jsonl', lines=True)
data_gramma_gemma = pd.read_json('data/data_raw/gramma2.jsonl', lines=True)
dat = [data_gramma_odin, data_gramma_gemma]
data_gramma = pd.concat(dat)
data_gramma = data_gramma.drop_duplicates()
data_gramma.to_json('data/data_processed/gramma.jsonl', orient='records', lines=True, force_ascii=False)

data_image_odin = pd.read_json('data/data_raw/image_odin.jsonl', lines=True)
data_image_gemma = pd.read_json('data/data_raw/image2.jsonl', lines=True)
dat = [data_image_gemma, data_image_odin]
data_image = pd.concat(dat)
data_image = data_image.drop_duplicates()
data_image.to_json('data/data_processed/image.jsonl', orient='records', lines=True, force_ascii=False)

data_speech_odin = pd.read_json('data/data_raw/speech_odin.jsonl', lines=True)
data_speech_gemma = pd.read_json('data/data_raw/speech2.jsonl', lines=True)
dat = [data_speech_gemma, data_speech_odin]
data_speech = pd.concat(dat)
data_speech = data_speech.drop_duplicates()
data_speech.to_json('data/data_processed/speech.jsonl', orient='records', lines=True, force_ascii=False)

data_web_odin = pd.read_json('data/data_raw/web_odin.jsonl', lines=True)
data_web_gemma = pd.read_json('data/data_raw/web2-2.jsonl', lines=True)
dat = [data_web_gemma, data_web_odin]
data_web = pd.concat(dat)
data_web = data_web.drop_duplicates()
data_web.to_json('data/data_processed/web.jsonl', orient='records', lines=True, force_ascii=False)