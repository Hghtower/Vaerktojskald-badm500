from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = torch.device("cuda")

tokenizer = AutoTokenizer.from_pretrained("google/t5gemma-2b-2b-prefixlm-it", token="")
model = AutoModelForSeq2SeqLM.from_pretrained("google/t5gemma-2b-2b-prefixlm-it", token="").to(device)


chat_template = '<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n'
prompt = chat_template.format(
    user_input='Generate 10 more examples like this:'
)

input_ids = tokenizer(prompt, return_tensors="pt").to(device)
output = model.generate(**input_ids, max_new_tokens=128)

print(tokenizer.decode(output[0], skip_special_tokens=True))



