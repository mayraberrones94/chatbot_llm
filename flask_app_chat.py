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
model_path="dream_gpt2/"

def load_model(model_path, device='cpu'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    model.to(device)

    return tokenizer, model

tokenizer, model = load_model(model_path)

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

        <label for="initial-prompt">Initial Question:</label>
        <input type="text" value="when did it begin exactly?" name="initial-prompt" />


            <label for="mode">Follow up Mode:</label>
            <select id="mode" name="mode">
                <option value="single">Single Prompt</option>
                <option value="multiple" selected>Multiple Prompts</option>
            </select>

            <!-- Single prompts container -->
            <div id="single-prompt-section" style="display: none; margin-top: 10px;">
                <label for="single-prompt">Prompt:</label>
                <textarea id="single-prompt" name="single-prompt" rows="4" cols="50"
                    placeholder="Enter your full prompt here..."></textarea>
            </div>

            <!-- Multiple prompts container -->
            <div id="multiple-prompts-section" style="margin-top: 10px;">
                <label for="claude-p1">Follow Up Prompt 1:</label>
                <textarea id="claude-p1" name="claude-p1" rows="4" cols="50"
                    placeholder="You are role playing a game of telephone, in which your task is to repeat a piece of text, indicated by <text></text> tags. Your role is to attempt to repeat the paragraph, though you misremember phrases, sometimes retaining their approximate meaning but changing the words, sometimes changing the meaning quite a bit but retaining some of the texture. Aim for a natural tone. Do NOT include any extra commentary or explanation."></textarea>

                <label for="claude-p2">Follow Up Prompt 2:</label>
                <textarea id="claude-p2" name="claude-p2" rows="4" cols="50"></textarea>

                <label for="claude-p3">Follow Up Prompt 3:</label>
                <textarea id="claude-p3" name="claude-p3" rows="4" cols="50"></textarea>
            </div>

            <button type="submit">submit</button>
        </form>
    </div>

    <h2>Results</h2>
    <div id="results">
    </div>

    <script>
        const container = document.getElementById("results");
        const form = document.getElementById("prompt-form");

        form.addEventListener("submit", function(e) {
            e.preventDefault(); // prevent page reload
            container.innerHTML = ""; // clear previous responses

            const mode = form.elements["mode"].value;
            let promptData;

            if (mode === "single") {
                // Single prompt mode: just take the textarea
                promptData = {
                    mode: "single",
                    initial: form.elements["initial-prompt"].value,
                    prompt: form.elements["single-prompt"].value
                };
            } else {
                // Multiple prompt mode: gather all inputs
                promptData = {
                    mode: "multiple",
                    initial: form.elements["initial-prompt"].value,
                    prompts: [
                        form.elements["claude-p1"].value,
                        form.elements["claude-p2"].value,
                        form.elements["claude-p3"].value
                    ]
                };
            }

            // Open SSE stream with JSON payload encoded in query string
            const evtSource = new EventSource(
                `/stream?data=${encodeURIComponent(JSON.stringify(promptData))}`
            );

            evtSource.onmessage = function(event) {
                const responses = JSON.parse(event.data);
                container.innerHTML = ""; // replace previous content
                responses.forEach((r) => {
                    const p = document.createElement("p");
                    p.textContent = r;
                    container.appendChild(p);
                });
            };

            evtSource.onerror = function() {
                evtSource.close(); // close stream on error
            };
        });

        const modeSelect = document.getElementById("mode");
        const singleSection = document.getElementById("single-prompt-section");
        const multipleSection = document.getElementById("multiple-prompts-section");

        function toggleSections() {
            if (modeSelect.value === "single") {
                singleSection.style.display = "block";
                multipleSection.style.display = "none";
            } else {
                singleSection.style.display = "none";
                multipleSection.style.display = "block";
            }
        }

        // Run on page load (in case default selection is not "multiple")
        toggleSections();

        // Listen for changes
        modeSelect.addEventListener("change", toggleSections);
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


def claude_follow_up(prompt, previous_text):
    msgs = []
    msgs.append({"role": "user", "content": f"<text>{previous_text}</text>"})

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
    raw_data = request.args.get("data")

    if raw_data:
        try:
            prompt_data = json.loads(raw_data)
        except json.JSONDecodeError:
            return "Invalid JSON in 'data' parameter", 400

    else:
        return "no data", 400

    def generate():
        responses = []
        initial = prompt_data.get("initial", "")
        responses.append(generate_initial_response(initial, tokenizer, model))
        yield f"data:{json.dumps(responses)}\n\n"


        if prompt_data.get("mode") == "single":
            prompt = prompt_data.get("prompt")
            responses.append(claude_follow_up(prompt, responses[0]))
            yield f"data:{json.dumps(responses)}\n\n"

        elif prompt_data.get("mode") == "multiple":
            prompts = prompt_data.get("prompts", [])

            for i, prompt in enumerate(prompts):
                responses.append(claude_follow_up(prompt, responses[i]))
                yield f"data:{json.dumps(responses)}\n\n"

        else:
            return "Unknown mode", 400

    return Response(generate(), mimetype="text/event-stream")


@app.route("/", methods=["GET"])
def index():
    return render_template_string(TEMPLATE)


if __name__ == "__main__":
    app.run(port=5050)
