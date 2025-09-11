from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, render_template, render_template_string, request, Response, jsonify
import torch
import sys
import time
import re
import os
import json
from dotenv import load_dotenv
import sqlite3
from claude_utils import call_model
import anthropic
import database_utils
load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
PASSWORD = "photocopier!"

def check_auth(password):
    """Check if the provided password is correct."""
    return password == PASSWORD

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
        'Could not verify your access level for that URL.\n'
        'You have to provide the correct password', 401,
        {'WWW-Authenticate': 'Basic realm="Login Required"'}
    )

def load_model(model_path, device='cpu'):
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model.eval()
    model.to(device)

    return tokenizer, model

dream_model_path="dream_gpt2/"
dream_tokenizer, dream_model = load_model(dream_model_path)

email_model_path="dream_email/"
email_tokenizer, email_model = load_model(email_model_path)


app = Flask(__name__)
app.app_context().push()
app.teardown_appcontext(database_utils.close_db)

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
    <div id="main">
        <div>
        <h2>Setup</h2>
        <div id="form-container">
            <form id="prompt-form" method="get">

            <label for="initial-prompt">Initial Question:</label>
            <input type="text" value="when did it begin exactly?" name="initial-prompt" />

                <label for="model">Model:</label>
                <select id="model" name="model">
                    <option value="dream_gpt2">Dream + GPT2</option>
                    <option value="dream_email" selected>Dream + Email + GPT2</option>
                </select>

                <label for="mode">Follow up Mode:</label>
                <select id="mode" name="mode">
                    <option value="single">Single Prompt</option>
                    <option value="multiple" selected>Multiple Prompts</option>
                </select>

                <!-- Single prompts container -->
                <div id="single-prompt-section" style="display: none; margin-top: 10px;">
                    <div class="prompt-wrapper">
                        <label for="single-prompt">Prompt:</label>
                        <textarea id="single-prompt" name="single-prompt" rows="4" cols="50"
                            placeholder="Enter your full prompt here..."></textarea>
                        <p> Note: the phrase "Separate each generated text with an underscore." is added onto this prompt on the backend to allow the single response of the model to be split into multiple phrases. You shouldn't need to add any additional separators. </p>
                    </div>
                </div>

                <!-- Multiple prompts container -->
                <div id="multiple-prompts-section" style="margin-top: 10px;">
                    <div id="follow-up">
                        <div class="prompt-wrapper">
                        <label for="claude-p1">Follow Up Prompt 1:</label>
                        <textarea id="claude-p1" name="claude-prompts" rows="4" cols="50"
                            placeholder="You are role playing a game of telephone, in which your task is to repeat a piece of text, indicated by <text></text> tags. Your role is to attempt to repeat the paragraph, though you misremember phrases, sometimes retaining their approximate meaning but changing the words, sometimes changing the meaning quite a bit but retaining some of the texture. Aim for a natural tone. Do NOT include any extra commentary or explanation."></textarea>
                        </div>
                    </div>
                    <button type="button" id="add-prompt-btn">+ Add Follow-Up Prompt</button>
                </div>


                <button type="submit">submit</button>
            </form>
        </div>
        </div>

        <div>
            <h2>Results</h2>
            <div id="results">
                <div id="initial"> </div>
                <form id="db-form">
                    <div id="db-form-inner">
                    </div>
                    <button type="submit">submit to database</button>
                </form>
            </div>
        </div>
    </div>
</body>
    <script src="/static/main.js"></script>
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
    final_res = ""

    while len(final_res) < 20:
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
        final_res = clean_output(response)

    return final_res

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
        model = dream_model
        tokenizer = dream_tokenizer

        if prompt_data.get("model") == "dream_gpt2":
            model = dream_model
            tokenizer = dream_tokenizer

        elif prompt_data.get("model") == "dream_email":
            model = email_model
            tokenizer = email_tokenizer

        else:
            return "Unknown model", 400

        responses.append(generate_initial_response(initial, tokenizer, model))
        yield f"data:{json.dumps(responses)}\n\n"

        if prompt_data.get("mode") == "single":
            prompt = prompt_data.get("prompt")
            res = claude_follow_up(prompt + " Separate each generated text with an underscore.", responses[0])
            res_list = res.split("_")
            responses = responses + res_list
            yield f"data:{json.dumps(responses)}\n\n"

        elif prompt_data.get("mode") == "multiple":
            prompts = prompt_data.get("prompts", [])

            for i, prompt in enumerate(prompts):
                responses.append(claude_follow_up(prompt, responses[i]))
                yield f"data:{json.dumps(responses)}\n\n"

        else:
            return "Unknown mode", 400

    return Response(generate(), mimetype="text/event-stream")


@app.route("/init")
def init():
    database_utils.init_db(app)
    return "Database initialized!"

@app.before_request
def require_password():
    auth = request.authorization
    if not auth or not check_auth(auth.password):
        return authenticate()

@app.route('/api/dialogues')
def get_dialogues():
    try:
        # Get database connection
        conn = sqlite3.connect('dialogues.db')  # Replace with your database path
        conn.row_factory = sqlite3.Row  # This allows dict-like access to rows
        cursor = conn.cursor()
        
        # Query to get all dialogues with their blocks
        cursor.execute("""
            SELECT 
                d.id,
                d.prompt,
                d.initial_response,
                d.created_at,
                db.speaker,
                db.block_order,
                db.text as block_text
            FROM dialogue d
            LEFT JOIN dialogue_block db ON d.id = db.dialogue_id
            ORDER BY d.id DESC, db.block_order ASC
        """)
        
        rows = cursor.fetchall()
        
        # Group blocks by dialogue
        dialogues = {}
        for row in rows:
            dialogue_id = row['id']
            
            if dialogue_id not in dialogues:
                dialogues[dialogue_id] = {
                    'id': dialogue_id,
                    'prompt': row['prompt'],
                    'initial_response': row['initial_response'],
                    'created_at': row['created_at'],
                    'blocks': []
                }
            
            # Add block if it exists
            if row['speaker']:
                dialogues[dialogue_id]['blocks'].append({
                    'speaker': row['speaker'],
                    'text': row['block_text'],
                    'order': row['block_order']
                })
        
        conn.close()
        
        # Convert to list and return as JSON
        dialogues_list = list(dialogues.values())
        return jsonify(dialogues_list)
        
    except Exception as e:
        print(f"Error fetching dialogues: {e}")
        return jsonify({'error': 'Failed to fetch dialogues'}), 500

@app.route('/api/dialogues/<int:dialogue_id>', methods=['DELETE'])
def delete_dialogue(dialogue_id):
    try:
        conn = sqlite3.connect("dialogues.db")
        cursor = conn.cursor()
        
        # Delete dialogue (blocks will be deleted automatically due to CASCADE)
        cursor.execute("DELETE FROM dialogue WHERE id = ?", (dialogue_id,))
        conn.commit()
        
        if cursor.rowcount == 0:
            conn.close()
            return jsonify({'error': 'Dialogue not found'}), 404
            
        conn.close()
        return jsonify({'success': True, 'message': 'Dialogue deleted'})
        
    except Exception as e:
        print(f"Error deleting dialogue: {e}")
        return jsonify({'error': 'Failed to delete dialogue'}), 500

@app.route('/api/dialogues/<int:dialogue_id>', methods=['PUT'])
def update_dialogue(dialogue_id):
    try:
        data = request.get_json()
        
        conn = sqlite3.connect("dialogues.db")
        cursor = conn.cursor()
        
        # Update dialogue
        cursor.execute("""
            UPDATE dialogue 
            SET prompt = ?, initial_response = ? 
            WHERE id = ?
        """, (data['prompt'], data['initial_response'], dialogue_id))
        
        # Delete existing blocks
        cursor.execute("DELETE FROM dialogue_block WHERE dialogue_id = ?", (dialogue_id,))
        
        # Insert updated blocks
        for i, block in enumerate(data['blocks']):
            cursor.execute("""
                INSERT INTO dialogue_block (dialogue_id, speaker, block_order, text)
                VALUES (?, ?, ?, ?)
            """, (dialogue_id, block['speaker'], i, block['text']))
        
        conn.commit()
        conn.close()
        
        return jsonify({'success': True, 'message': 'Dialogue updated'})
        
    except Exception as e:
        print(f"Error updating dialogue: {e}")
        return jsonify({'error': 'Failed to update dialogue'}), 500


@app.route("/insert")
def example():
    raw_data = request.args.get("data")
    print("got req, raw data is", raw_data)
    # Parse the JSON data
    data = json.loads(raw_data)
    
    # Insert directly into database
    dialogue_id = database_utils.insert_dialogue(
        prompt=data['prompt'],
        initial_response=data['initial_response'],
        blocks=data['blocks']
    )

    return f"Inserted dialogue {dialogue_id}"

@app.route('/history')
def history():
    return render_template('history.html')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5050)
