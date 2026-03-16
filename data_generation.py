from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from vllm import LLM, SamplingParams


device = torch.device("cuda")

# tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-4b-it-unsloth-bnb-4bit", token="")
# model = AutoModelForSeq2SeqLM.from_pretrained("unsloth/gemma-3-4b-it-unsloth-bnb-4bit", token="").to(device)

# chat_template = '<start_of_turn>user\n{user_input}<end_of_turn>\n<start_of_turn>model\n'
# prompt = chat_template.format(
#     user_input='Generate 10 more examples like this:'
# )

# input_ids = tokenizer(prompt, return_tensors="pt").to(device)
# output = model.generate(**input_ids, max_new_tokens=128)

# print(tokenizer.decode(output[0], skip_special_tokens=True))


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

def main():
    # Create an LLM.
    llm = LLM(model="facebook/opt-125m")
    # Generate texts from the prompts.
    # The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()