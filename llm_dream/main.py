# Main code for llm

from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import sys
from gen_responses import *

def load_model(model_path):
    #Check device, and if available use cuda or mps. Default is cpu
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    model.to(device)
    return tokenizer, model, device

'''
(agnes_llm) mberrones@mberrones-Precision-5820-Tower-X-Series:~/Documents/llm_dream$ python main.py gpt2_claude_finetuned/ responses_followups.csv response_updated.csv
'''
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python chatbot.py <model_path>")
        sys.exit(1)
    model_path = sys.argv[1] #This accepts the whole folder with the output of the training from gpt
    mode = sys.argv[2]
    tokenizer, model, device = load_model(model_path)
    if mode == 'user':
        print("Chatbot is ready! Type 'exit' to quit.\n")
        run_query_user_mode(tokenizer, model, device)
    else:
        run_text_queries(tokenizer, model, device)


