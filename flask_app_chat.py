from transformers import GPT2Tokenizer, GPT2LMHeadModel
from flask import Flask, render_template, render_template_string, request, Response, jsonify, send_file, redirect, url_for, abort
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
import audio_utils
from elevenlabs import stream, save
from datetime import datetime
load_dotenv()

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
PASSWORD = "photocopier!"
AUDIO_DIR = "audio"

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


@app.before_request
def require_password():
    auth = request.authorization
    if not auth or not check_auth(auth.password):
        return authenticate()

@app.route('/api/dialogues')
def get_dialogues():
    try:
        conn = sqlite3.connect('dialogues.db') 
        conn.row_factory = sqlite3.Row 
        cursor = conn.cursor()
        
        # need to add a line to see if audio generated

        cursor.execute("""
            SELECT 
                d.id,
                d.prompt,
                d.initial_response,
                d.created_at,
                db.speaker,
                db.block_order,
                db.text as block_text,
                da.created_at as audio_created_at,  -- Add this!
                da.file_path as audio_file_path     -- And maybe this
            FROM dialogue d
            LEFT JOIN dialogue_block db ON d.id = db.dialogue_id
            LEFT JOIN dialogue_audio da ON db.id = da.block_id  -- Keep LEFT JOIN
            ORDER BY d.id DESC, db.block_order ASC;
        """)
        
        rows = cursor.fetchall()
        
        # Group blocks by dialogue
        dialogues = {}
        for row in rows:
            dialogue_id = row['id']
            audio = True if row['audio_created_at'] else False

            if dialogue_id not in dialogues:
                dialogues[dialogue_id] = {
                    'id': dialogue_id,
                    'prompt': row['prompt'],
                    'initial_response': row['initial_response'],
                    'created_at': row['created_at'],
                    'blocks': [],
                    'audio': audio
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

@app.route("/generate-audio", methods=["GET", "POST"])
def generate_audio():
    db = database_utils.get_db()
    cur = db.cursor()

    stability = request.form.get("stability")
    use_speaker_boost = request.form.get("use_speaker_boost") == "on"
    similarity_boost = request.form.get("similarity_boost")
    style = request.form.get("style")
    speed = request.form.get("speed")

    # Convert optional values to proper types or None
    params = {
        "stability": float(stability) if stability else None,
        "use_speaker_boost": use_speaker_boost,
        "similarity_boost": float(similarity_boost) if similarity_boost else None,
        "style": float(style) if style else None,
        "speed": float(speed) if speed else None,
    }
    
    if request.method == "POST":
        # Generate audio and get the dialogue_id
        dialogue_id = audio_utils.audio_gen(request, db, cur, params)
        
        # Fetch the generated audio files for this dialogue
        cur.execute("""
            SELECT da.block_id, da.speaker, da.voice, da.file_path, db.block_order, db.text
            FROM dialogue_audio da
            JOIN dialogue_block db ON da.block_id = db.id
            WHERE db.dialogue_id = ?
            ORDER BY db.block_order
        """, (dialogue_id,))
        
        audio_files = cur.fetchall()
        
        # Return JSON response with audio file info
        return jsonify({
            'success': True,
            'dialogue_id': dialogue_id,
            'audio_files': [
                {
                    'block_id': row[0],
                    'speaker': row[1],
                    'voice': row[2],
                    'file_path': row[3],
                    'block_order': row[4],
                    'text': row[5]
                }
                for row in audio_files
            ]
        })
    
    # GET request - render form
    cur.execute("SELECT id FROM dialogue ORDER BY id")
    dialogues = cur.fetchall()
    return render_template("generate_audio.html", dialogues=dialogues)

# New route to serve audio files
@app.route("/audio/<path:filename>")
def serve_audio(filename):
    # Security: ensure the file is within AUDIO_DIR
    safe_path = os.path.join(filename)
    if not os.path.commonpath([AUDIO_DIR, safe_path]) == AUDIO_DIR:
        abort(403)
    
    return send_file(safe_path, mimetype="audio/mpeg")

@app.route("/insert")
def example():
    raw_data = request.args.get("data")
    # Parse the JSON data
    data = json.loads(raw_data)
    
    # Insert directly into database
    dialogue_id = database_utils.insert_dialogue(
        prompt=data['prompt'],
        initial_response=data['initial_response'],
        blocks=data['blocks']
    )

    return f"Inserted dialogue {dialogue_id}"

@app.route("/dialogue/<int:dialogue_id>", methods=["GET", "POST"])
def view_dialogue(dialogue_id):
    db = database_utils.get_db()
    cur = db.cursor()
    timestamp = int(datetime.utcnow().timestamp())

    if request.method == "POST":
        block_id = int(request.form["block_id"])

        new_text = request.form["text"].strip()
        voice = request.form["voice"]
        speaker = request.form["speaker"]

        stability = request.form.get("stability")
        use_speaker_boost = request.form.get("use_speaker_boost") == "on"
        similarity_boost = request.form.get("similarity_boost")
        style = request.form.get("style")
        speed = request.form.get("speed")

        # Convert optional values to proper types or None
        params = {
            "stability": float(stability) if stability else None,
            "use_speaker_boost": use_speaker_boost,
            "similarity_boost": float(similarity_boost) if similarity_boost else None,
            "style": float(style) if style else None,
            "speed": float(speed) if speed else None,
        }


        # Update text
        cur.execute("UPDATE dialogue_block SET text = ? WHERE id = ?", (new_text, block_id))

        # Generate audio
        audio = audio_utils.call_elevenlabs_api(new_text, voice, params)

        dialogue_dir = os.path.join("audio", f"dialogue_{dialogue_id}")
        os.makedirs(dialogue_dir, exist_ok=True)

        filename = f"block_{block_id}_{speaker}.mp3"
        file_path = os.path.join(dialogue_dir, filename)
        save(audio, file_path)

        # Record in DB
        cur.execute("""
        INSERT INTO dialogue_audio
          (block_id, speaker, voice, file_path, stability, use_speaker_boost, similarity_boost, style, speed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            block_id,
            speaker,
            voice,
            file_path,
            stability if stability is not None else None,
            1 if use_speaker_boost else 0,
            similarity_boost if similarity_boost is not None else None,
            style if style is not None else None,
            speed if speed is not None else None,
        ))
        db.commit()

        return redirect(url_for("view_dialogue", dialogue_id=dialogue_id, timestamp=timestamp))

    # Fetch dialogue blocks + latest audio
    cur.execute(
        """
        SELECT b.id, b.speaker, b.block_order, b.text, a.file_path, a.voice
        FROM dialogue_block b
        LEFT JOIN (
            SELECT da.block_id, da.file_path, da.voice
            FROM dialogue_audio da
            JOIN (
                SELECT block_id, MAX(created_at) AS latest
                FROM dialogue_audio
                GROUP BY block_id
            ) latest ON da.block_id = latest.block_id AND da.created_at = latest.latest
        ) a ON b.id = a.block_id
        WHERE b.dialogue_id = ?
        ORDER BY b.block_order
        """,
        (dialogue_id,),
    )
    blocks = cur.fetchall()

    return render_template("view_dialogue.html", dialogue_id=dialogue_id, blocks=blocks, timestamp=timestamp)


@app.route('/history')
def history():
    return render_template('history.html')

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(port=5050)
