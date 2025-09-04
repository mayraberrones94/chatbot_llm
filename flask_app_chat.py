from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, render_template_string, request, Response
import torch
import sys
import time
import re
import os
import json
from dotenv import load_dotenv
from claude_utils import call_model
import anthropic
load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

app = Flask(__name__)
app.app_context().push()

# Template
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Prompt Testing</title>
    <link rel="stylesheet" href="/static/style.css" />
</head>
<body>
    <h1>Prompt Testing</h1>
    <div id="form-container">
        <form id="prompt-form" method="get">
            <label>Prompt:</label>
            <input type="text" value="prompt" name="prompt" />
            </select>
            <button type="submit">submit</button>
        </form>
    </div>

    <div id="results">
        <h2>Results</h2>
    </div>

    <script>
        const container = document.getElementById("results");
        const form = document.getElementById("prompt-form");

        form.addEventListener("submit", function(e) {
            e.preventDefault(); // prevent page reload
            container.innerHTML = ""; // clear previous responses

            const prompt = form.elements["prompt"].value;
            const evtSource = new EventSource(`/stream?prompt=${encodeURIComponent(prompt)}`);

            evtSource.onmessage = function(event) {
                const responses = JSON.parse(event.data);
                container.innerHTML = ""; // replace previous content
                responses.forEach((r, i) => {
                    const p = document.createElement("p");
                    p.textContent = `${r}`;
                    container.appendChild(p);
                });
            };

            evtSource.onerror = function() {
                evtSource.close(); // close stream on error
            };
        });
    </script>
</body>
</html>
"""

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

@app.route("/stream")
def stream():
    user_input = request.args.get("prompt", "when did it begin exactly?")

    def generate():
        responses = []

        # Step 1
        responses.append(generate_initial_response(user_input, tokenizer, model))
        yield f"data:{json.dumps(responses)}\n\n"

        # Step 2
        responses.append(claude_follow_up(responses[0]))
        yield f"data:{json.dumps(responses)}\n\n"

        # Step 3
        responses.append(claude_follow_up(responses[1]))
        yield f"data:{json.dumps(responses)}\n\n"

        # Step 4
        responses.append(claude_follow_up(responses[2]))
        yield f"data:{json.dumps(responses)}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE)


if __name__ == "__main__":
    model_path="dream_gpt2/"
    tokenizer, model = load_model(model_path)

    app.run(port=5050)
