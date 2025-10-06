import whisper
import pyaudio
import wave
import tempfile
import time

# Load the Whisper model once
model = whisper.load_model("base.en")

def transcribe_directly():
    # Create a temporary file to store recorded audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        wav_path = temp_file.name

    sample_rate = 16000
    bits_per_sample = 16
    chunk_size = 1024
    audio_format = pyaudio.paInt16
    channels = 1

    audio = pyaudio.PyAudio()

    # Open output wave file
    wav_file = wave.open(wav_path, 'wb')
    wav_file.setnchannels(channels)
    wav_file.setsampwidth(audio.get_sample_size(audio_format))
    wav_file.setframerate(sample_rate)

    # Open input stream
    stream = audio.open(format=audio_format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

    print("Recording ... (press Ctrl+C to exit)\n")
    start_time = time.time()

    try:
        while True:
            data = stream.read(chunk_size, exception_on_overflow=False)
            wav_file.writeframes(data)
            if time.time() - start_time >= 15:
                break
    except KeyboardInterrupt:
        print("\nRecording stopped early...")

    # Clean up
    stream.stop_stream()
    stream.close()
    audio.terminate()
    wav_file.close()

    # Transcribe using Whisper
    print("Transcribing...")
    result = model.transcribe(wav_path, fp16=False)

    text = result["text"].strip()
    print("\nText:\n")
    print(text)

    return text
'''
if __name__ == "__main__":
    transcribe_directly()
'''

