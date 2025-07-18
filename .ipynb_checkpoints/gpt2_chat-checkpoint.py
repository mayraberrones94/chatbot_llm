from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys



def load_model(model_path):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model.eval()
    return tokenizer, model

def generate_response(prompt, tokenizer, model, max_length=150):
    full_prompt = f"User: {prompt}\nBot:"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")

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
        )

    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the part after "Bot:"
    if "Bot:" in decoded_output:
        response = decoded_output.split("Bot:")[-1].strip()
    else:
        response = decoded_output[len(prompt):].strip()

    return response

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
