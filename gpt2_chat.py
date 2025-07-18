from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
import re


def extract_whole_sentences(text):
    # Match sentences that:
    # - Start with a capital letter
    # - Have some characters (non-greedy)
    # - End with punctuation (., !, or ?)
    pattern = r'\b[A-Z][^.!?]{3,200}?[.!?](?=\s|$)'
    return re.findall(pattern, text)

def clean_output(text):
    # get rid of html, only keep whole sentences
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[#A-Za-z0-9]+;', '', text)
    text = text.strip()
    sentences = extract_whole_sentences(text)
    return " ".join(sentences)


def load_model(model_path, device='cpu'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    model.to(device)

    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=150):
    full_prompt = f"You are role playing a conversation. You should respond to questions in full sentences, in a paragraph between 100-200 words in length. Do not include descriptive or categorical phrases, like 'Settings and Characters', 'Interpretation' or 'Feelings and Thoughts'. Do not include references or information in brackets. Question: {prompt}\n"
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

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <model_path>")
        sys.exit(1)

    model_path = sys.argv[1] #This accepts the whole folder with the output of the training from gpt
    tokenizer, model = load_model(model_path)

    print("Chatbot is ready! Type 'exit' to quit.\n")

    while True:
        user_input = input("User: ") #Ctrl + C for emergencies
        if user_input.lower() in ["exit", "quit"]:
            break
        response = generate_response(user_input, tokenizer, model)
        print("Bot:", response)
