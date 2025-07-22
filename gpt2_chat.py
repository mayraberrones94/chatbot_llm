from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
import re
import os
from dotenv import load_dotenv
from claude_utils import call_model
import anthropic
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def trim_fragment_start(text):
    # Step 1: Remove leading fragment like "October).", "Europe)."
    text = re.sub(r'^[A-Z][a-z]+\)\.\s*', '', text)

    # Step 2: Find first sentence-like capitalized word, skipping junk
    match = re.search(r'\b[A-Z][a-z]+.*', text)

    if match:
        return text[match.start():].strip()
    else:
        return text.strip()

def extract_whole_sentences(text):
    # Match sentences that:
    # - Start with a capital letter
    # - Have some characters (non-greedy)
    # - End with punctuation (., !, or ?)
    pattern = r'\b[A-Z][^.!?]{3,200}?[.!?](?=\s|$)'
    return re.findall(pattern, text)

def clean_output(text):
    # Remove HTML tags like <p>, <br>, etc.
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove HTML entities like &nbsp;, &#39;, etc.
    text = re.sub(r'&[#A-Za-z0-9]+;', '', text)
    
    # Remove quotation marks
    text = re.sub(r'[\"“”]', '', text)
    text = text.strip()
    sentences = extract_whole_sentences(text)

    # Rejoin and trim leading fragments
    new_text = " ".join(sentences)
    new_text = trim_fragment_start(new_text)

    return new_text.strip().lstrip()

def load_model(model_path, device='cpu'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    model.to(device)

    return tokenizer, model

# this is a call to the dream model
def generate_follow_up(prompt, tokenizer, model, max_length=200):
    full_prompt = f"We are playing a game of telephone -- please attempt to repeat the text that appears in the <text> tags. Do NOT include any explanations, dialogue, or additional text: <text>{prompt}</text>.\nRepeat now:"
    
    # print('full prompt is', full_prompt)
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            repetition_penalty=1.2
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the part after "Bot:"
    if "Bot:" in decoded_output:
        response = decoded_output.split("Bot:")[-1].strip()
    else:
        response = decoded_output[len(full_prompt):].strip()

    return clean_output(response)


def claude_follow_up(prompt):
    msgs = []
    msgs.append({"role": "user", "content": f"<text>{prompt}</text>"})
    prompt = "You are role playing a game of telephone, in which your task is to repeat a piece of text, indicated by <text></text> tags. Your role is to attempt to repeat the paragraph, though you misremember phrases, sometimes retaining their approximate meaning but changing the words, sometimes changing the meaning quite a bit but retaining some of the texture. Aim for a natural tone. Do NOT include any extra commentary or explanation."


    outcome = call_model(
        client, 
        msgs, 
        prompt, 
        temperature=1
    )

    return outcome.content[0].text


def claude_rhyming_follow_up(prompt):
    msgs = []
    msgs.append({"role": "user", "content": f"<text>{prompt}</text>"})
    prompt = "You are role playing a game of telephone, in which your task is to repeat a piece of text, indicated by <text></text> tags. In this instance, make your response rhyming."

    outcome = call_model(
        client, 
        msgs, 
        prompt, 
        temperature=1
    )

    return outcome


def generate_initial_response(prompt, tokenizer, model, max_length=150):
    # full_prompt = f"You are role playing a conversation. You should respond to questions in full sentences, in a paragraph between 100-200 words in length. Do not include descriptive or categorical phrases, like 'Settings and Characters', 'Interpretation' or 'Feelings and Thoughts'. Do not include references or information in brackets. Question: {prompt}\n"
    full_prompt = f"Question: {prompt} Bot:\n"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.4,
            repetition_penalty=1.5
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the part after "Bot:"
    if "Bot:" in decoded_output:
        response = decoded_output.split("Bot:")[-1].strip()
    else:
        response = decoded_output[len(full_prompt):].strip()

    return clean_output(response)


def run_text_queries():
    with open('questions.txt') as questions:
        for question in questions:
            print("\nquestion:", question)

            initial_response = generate_initial_response(question, tokenizer, model)
            print("\ninitial response:", initial_response)

            # generate the follow up using the dreambank dataset
            follow_1 = generate_follow_up(initial_response, tokenizer, model)
            print("\ndream dataset follow up:", follow_1)

            # generate the follow up using claude api
            follow_2 = claude_follow_up(initial_response)
            print("\nclaude initial follow up:", follow_2)
            follow_3 = claude_follow_up(follow_2)
            print("\nclaude second follow up:", follow_3)
            follow_4 = claude_follow_up(follow_3)
            print("\nclaude third follow up:", follow_4)


def run_query_user_mode():
    while True:
        user_input = input("User: ") #Ctrl + C for emergencies
        if user_input.lower() in ["exit", "quit"]:
            break
        initial_response = generate_initial_response(user_input, tokenizer, model)
        print("\ninitial response:", initial_response)

        # generate the follow up using the dreambank dataset
        follow_1 = generate_follow_up(initial_response, tokenizer, model)
        print("\ndream dataset follow up:", follow_1)

        # generate the follow up using claude api
        follow_2 = claude_follow_up(initial_response)
        print("\nclaude initial follow up:", follow_2)
        follow_3 = claude_follow_up(follow_2)
        print("\nclaude second follow up:", follow_3)
        follow_4 = claude_follow_up(follow_3)
        print("\nclaude third follow up:", follow_4)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1] #This accepts the whole folder with the output of the training from gpt
    mode = sys.argv[2]
    tokenizer, model = load_model(model_path)

    if mode == 'user':
        print("Chatbot is ready! Type 'exit' to quit.\n")
        run_query_user_mode()

    else:
        run_text_queries()


