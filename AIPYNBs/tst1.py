# requirements:
# pip install openai-whisper pydub torch

import whisper
from pydub import AudioSegment

# ---- Step 1: Convert OGG to WAV ----
ogg_file = "WhatsApp_Ptt.ogg"   # <-- replace with your file
wav_file = "output.wav"

audio = AudioSegment.from_ogg(ogg_file)
audio.export(wav_file, format="wav")

# ---- Step 2: Load Whisper model ----
# "small" is faster, "medium" or "large" is more accurate
model = whisper.load_model("small")

# ---- Step 3: Transcribe Chinese ----
result = model.transcribe(wav_file, language="zh")

print("Chinese transcription:")
print(result["text"])

# ---- Step 4: Translate to English ----
# Whisper can directly translate too
translated = model.transcribe(wav_file, task="translate")

print("\nEnglish translation:")
print(translated["text"])
