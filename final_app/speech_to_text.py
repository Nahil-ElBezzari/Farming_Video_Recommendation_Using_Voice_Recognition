import os
import wave
import threading
import tkinter as tk
from tkinter import messagebox
import pyaudio
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account

# Google Speech client

credentials = service_account.Credentials.from_service_account_file('cle.json')
client = speech.SpeechClient(credentials=credentials)

recording = False
frames = []
output_filename = "temp_audio.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024


def record_audio():
    global recording, frames
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    transcribe_audio()


def toggle_recording():
    global recording
    if not recording:
        recording = True
        button.config(text="Stop Recording")
        threading.Thread(target=record_audio).start()
    else:
        recording = False
        button.config(text="Start Recording")


def transcribe_audio():
    try:
        with open(output_filename, "rb") as audio_file:
            content = audio_file.read()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="fr-FR"
        )

        audio = speech.RecognitionAudio(content=content)

        response = client.recognize(config=config, audio=audio)

        output = ""
        for result in response.results:
            output += f"Transcript: {result.alternatives[0].transcript}\n"
            output += f"Confidence: {result.alternatives[0].confidence:.2f}\n"

        messagebox.showinfo("Transcription", output)

    except Exception as e:
        messagebox.showerror("Erreur", str(e))
    finally:
        if os.path.exists(output_filename):
            os.remove(output_filename)


window = tk.Tk()
window.title("Enregistreur vocal")
button = tk.Button(window, text="Start Recording", command=toggle_recording, width=30, height=3)
button.pack(pady=20)
window.mainloop()