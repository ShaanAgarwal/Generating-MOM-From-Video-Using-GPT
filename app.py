import whisper
import openai
import os
os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"
import sys
import subprocess
from fastapi import FastAPI, File, UploadFile #to create a simple API
import aiofiles
import json

model = whisper.load_model("base")

openai.api_key = ""

app = FastAPI()

def video_to_audio(video_file):
    audio_file = "input_audio.mp3"
    ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
    cmd = [ffmpeg_path, "-y", "-i", video_file, audio_file]
    print("Running command:", " ".join(cmd))
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return audio_file

def audio_to_transcript(audio_file):
    result = model.transcribe(audio_file)
    transcript = result["text"]
    return transcript

def MoM_generation(prompt):
    prompt = "Can you generate the Minute Of Meeting In form of bullet points for the below transcript?\n" + prompt
    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response = response.choices[0].message.content
    return response

@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    filename = file.filename
    async with aiofiles.open(filename, mode = "wb") as f:
        await f.write(await file.read())
    
    audio_file = video_to_audio("interview.mp4")
    transcript = audio_to_transcript(audio_file)
    final_result = MoM_generation(transcript)
    response_body = final_result.replace("\n", " ")
    response_dict = {"response": response_body}
    json_result = json.dumps(response_dict)

    return json_result