import sys
import argparse
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import os

'''
This is where you tune:
temperature = Higher = more randomness.
top_p = Lower = limit to top % of probability mass.
top_k = Limit to top K most likely tokens.
repetition_penalty = Higher = less repeating.
max_length = How long the output can be.
no_repeat_ngram_size = Dont repeat phrases of this length.
do_sample = If True, it samples tokens instead of taking argmax.
'''



GENERATION_CONFIG = {
    "max_new_tokens": 300,
    "temperature": 1.7,
    "top_p": 1.0,
    "top_k": 500,
    "repetition_penalty": 0.9,
    "no_repeat_ngram_size": 0,
    "do_sample": True
}


def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer

def generate_response(model, tokenizer, prompt, config):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_length=input_ids.shape[1] + config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"],
            top_k=config["top_k"],
            repetition_penalty=config["repetition_penalty"],
            no_repeat_ngram_size=config["no_repeat_ngram_size"],
            do_sample=config["do_sample"]
        )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response[len(prompt):].strip()

def save_interaction(prompt, response, output_file):
    if output_file:
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(f" Prompt: {prompt}\n")
            f.write(f" Response: {response}\n")
            f.write("=" * 60 + "\n")

def run_chat(model_path, prompt_file=None, output_file=None):
    model, tokenizer = load_model(model_path)
    print(f" Loaded model from {model_path}")

    if prompt_file:
        print(f" Running prompts from {prompt_file}")
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompts = f.readlines()

        for i, prompt in enumerate(prompts, start=1):
            prompt = prompt.strip()
            if not prompt:
                continue
            response = generate_response(model, tokenizer, prompt, GENERATION_CONFIG)
            print(f"\n Prompt {i}: {prompt}")
            print(f" Response {i}: {response}")
            save_interaction(prompt, response, output_file)
    else:
        print(" Manual mode. Type your prompts (type 'exit' to quit).")
        while True:
            prompt = input("You: ")
            if prompt.lower() in {"exit", "quit"}:
                break
            response = generate_response(model, tokenizer, prompt, GENERATION_CONFIG)
            print(f"DreamBot: {response}")
            save_interaction(prompt, response, output_file)

# 🔌 Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_path", type=str, default="./dream_gpt2",
        help="Path to your fine-tuned GPT-2 model")
    parser.add_argument("--prompt_file", type=str, default=None,
        help="Optional path to .txt file with prompts")
    parser.add_argument("--output_file", type=str, default=None,
        help="Optional file path to save all interactions")
    args = parser.parse_args()
    run_chat(args.model_path, args.prompt_file, args.output_file)
