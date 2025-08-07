# generation of resposes

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
import re
import os
from clean_text import *

def generate_initial_response(prompt, tokenizer, model, device, max_length=150):
    full_prompt = f"Question: {prompt} Bot:\n"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            repetition_penalty=1.2
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Bot:" in decoded:
        return clean_output(decoded.split("Bot:")[-1])
    return clean_output(decoded[len(full_prompt):])

def generate_follow_up(prompt, tokenizer, model, device, max_length=200):
    full_prompt = (f"We are playing a game of telephone -- please attempt to repeat the text that appears in the <text>: <text>{prompt}</text>.")

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=1.0,
            top_k=100,
            top_p=0.95,
            repetition_penalty=1.1
        )
    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    if "Repeat now:" in decoded:
        return clean_output(decoded.split("Repeat now:")[-1])
    return clean_output(decoded[len(full_prompt):])

def run_query_user_mode(tokenizer, model, device):
    print("type 'exit' to quit.\n")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_initial_response(user_input, tokenizer, model, device)
        print(f"Response: {response}")
        follow_1 = generate_follow_up(response, tokenizer, model, device)
        print(f"Follow-up 1: {follow_1}")
        follow_2 = generate_follow_up(follow_1, tokenizer, model, device)
        print(f"Follow-up 2: {follow_2}")
        follow_3 = generate_follow_up(follow_2, tokenizer, model, device)
        print(f"Follow-up 3: {follow_3}")

def run_text_queries(tokenizer, model, device):
    with open('questions.txt') as questions:
        for question in questions:
            print("\nquestion:", question)

            initial_response = generate_initial_response(question, tokenizer, model, device)
            print("\ninitial response:", initial_response)

            # generate the follow up using the dreambank dataset
            follow_1 = generate_follow_up(initial_response, tokenizer, model)
            print("\ndream dataset follow up:", follow_1)


