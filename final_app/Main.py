import os
import wave
import threading
import tkinter as tk
import webbrowser
import pyaudio
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from google.cloud import speech_v1 as speech
from google.oauth2 import service_account
from neural1 import VideoClassifier, load_model

# Config Google Cloud
credentials = service_account.Credentials.from_service_account_file('cle.json')
client = speech.SpeechClient(credentials=credentials)

# Config audio
recording = False
frames = []
output_filename = "temp_audio.wav"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Modèle & données
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
videos_df = pd.read_csv("videos-question-form.csv", sep=";")
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
model, config, label_encoder = load_model(VideoClassifier, "video_classifier_model.pth")
model = model.to(device)

#Enregistrement audio
def record_audio():
    global recording, frames
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []
    while recording:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()

    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    transcribe_audio()

def toggle_recording():
    global recording
    if not recording:
        transcription_label.config(text="Transcription en cours...")
        result_label.config(text="")
        for widget in results_frame.winfo_children():
            widget.destroy()

        recording = True
        record_btn.config(text="Stop Recording")
        threading.Thread(target=record_audio).start()
    else:
        recording = False
        record_btn.config(text="Start Recording")

# Transcription + Recommandation
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

        transcription = ""
        for result in response.results:
            transcription += result.alternatives[0].transcript.strip()

        if transcription:
            display_recommendations(transcription)
        else:
            transcription_label.config(text="Rien n'a été reconnu. Essayez à nouveau.")
            result_label.config(text="")

    except Exception as e:
        result_label.config(text="Erreur : " + str(e))
    finally:
        if os.path.exists(output_filename):
            os.remove(output_filename)

# Affichage des recommandations
def display_recommendations(question):
    embedding = sentence_model.encode([question])
    tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        _, top_idx = torch.topk(probs, k=5)

    top_indices = top_idx[0].cpu().numpy()
    pred_ids = label_encoder.inverse_transform(top_indices)

    # Construire les résultats dans l’ordre des prédictions
    ordered_results = []
    for vid in pred_ids:
        row = videos_df[videos_df["id"] == vid][["titre", "url"]]
        if not row.empty:
            ordered_results.append(row.iloc[0])

    transcription_label.config(text=f"🗣️ Vous avez dit : « {question} »")
    result_label.config(text="Recommandations :")

    for widget in results_frame.winfo_children():
        widget.destroy()

    for i, row in enumerate(ordered_results):
        title = tk.Label(results_frame, text=row['titre'], wraplength=400, anchor="w")
        title.grid(row=i, column=0, sticky="w", padx=5, pady=5)

        btn = tk.Button(results_frame, text="Voir", command=lambda url=row['url']: webbrowser.open(url))
        btn.grid(row=i, column=1, padx=5)

# Interface Tkinter
window = tk.Tk()
window.title("Recommandation Vidéo Vocale")
window.geometry("700x500")

record_btn = tk.Button(window, text="Start Recording", command=toggle_recording, width=30, height=2)
record_btn.pack(pady=10)

transcription_label = tk.Label(window, text="", wraplength=650, justify="left", font=("Arial", 12, "italic"))
transcription_label.pack(pady=5)

result_label = tk.Label(window, text="Appuyez sur le bouton pour poser votre question.", wraplength=650, justify="left")
result_label.pack(pady=5)

results_frame = tk.Frame(window)
results_frame.pack(pady=10)

window.mainloop()
