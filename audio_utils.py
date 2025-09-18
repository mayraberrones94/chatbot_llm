import os
import requests
from flask import Flask, request, render_template, redirect, url_for
import database_utils
from elevenlabs import stream, save
from elevenlabs.client import ElevenLabs

app = Flask(__name__)

AUDIO_DIR = "audio"
FISH_API_URL = "https://api.fish.audio/v1/tts"  # adjust to real endpoint
FISH_API_KEY=os.environ["FISH_SPEECH_API_KEY"]
ELEVENLABS_API_KEY=os.environ["ELEVENLABS_API_KEY"]

elevenlabs = ElevenLabs(
  api_key=ELEVENLABS_API_KEY,
)

def call_elevenlabs_api(text, voice):
    ref_id = "sherron_id" if voice == "sherron" else "james_id"

    # Generate and save directly
    audio = elevenlabs.text_to_speech.convert(
        text=text,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )

    return audio

def call_fish_api(text, voice):
    ref_id = "8a54d1efde7f4d74abeaa13a9c39d33a" if voice == "sherron" else "701cda524ab24ab9a6ea2a1e844c2650"
    print('voice is', voice)

    resp = requests.post(
        FISH_API_URL,
        headers = {
            'Authorization': 'Bearer ' + FISH_API_KEY
        },
        json={
            "text": text,
            "model": "s1",
            "reference_id": ref_id,
            "format": "mp3",
            "sample_rate": 44100,
            "prosody": {
                "speed": 0.8,  # 20% faster (range: 0.5-2.0)
                "volume": 3    # Louder (range: -20 to 20)
            }
        },
    )
    resp.raise_for_status()
    return resp.content


def audio_gen(request, db, cur):
    dialogue_id = int(request.form["dialogue_id"])
    speaker1_voice = request.form["speaker1_voice"]
    speaker2_voice = request.form["speaker2_voice"]

    # Fetch dialogue blocks
    cur.execute(
        "SELECT id, speaker, block_order, text FROM dialogue_block WHERE dialogue_id = ? ORDER BY block_order",
        (dialogue_id,),
    )
    blocks = cur.fetchall()

    dialogue_dir = os.path.join(AUDIO_DIR, f"dialogue_{dialogue_id}")
    os.makedirs(dialogue_dir, exist_ok=True)

    for block in blocks:
        block_id, speaker, order, text = block
        if not text.strip():
            continue

        # Pick voice
        voice = speaker1_voice if speaker == "1" else speaker2_voice

        # Call elevenlabs API
        audio = call_elevenlabs_api(text, "voice")
        filename = f"block_{order}_{speaker}.mp3"
        file_path = os.path.join(dialogue_dir, filename)
        save(audio, file_path)

        # Delete existing entry for this specific block/speaker
        cur.execute(
            "DELETE FROM dialogue_audio WHERE block_id = ? AND speaker = ?",
            (block_id, speaker)
        )
        
        # Insert new entry
        cur.execute(
            "INSERT INTO dialogue_audio (block_id, speaker, voice, file_path) VALUES (?, ?, ?, ?)",
            (block_id, speaker, voice, file_path)
        )

    db.commit()
    
    # Return the dialogue_id instead of dialogues list
    return dialogue_id