import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from neural1 import VideoClassifier, load_model

# ===== Config & Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Chargement des données =====
videos_df = pd.read_csv("../../../Data/videos-question-form.csv", sep=";")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les labels pour aligner les IDs
questions_df = pd.read_csv("../../../Data/trainingSet.csv", sep=";")
label_encoder = LabelEncoder()
label_encoder.fit(questions_df["id"].tolist())

# ===== Charger le modèle entraîné proprement =====
model, config = load_model(VideoClassifier, "video_classifier_model.pth")
model = model.to(device)

# ===== Fonction de recommandation =====
def recommander_video(question, top_k=3):
    embedding = sentence_model.encode([question])
    tensor = torch.tensor(embedding, dtype=torch.float32).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        top_vals, top_idx = torch.topk(probs, k=top_k)

    pred_ids = label_encoder.inverse_transform(top_idx[0].cpu().numpy())
    pred_ids = label_encoder.inverse_transform(top_idx[0].cpu().numpy())

    # Filtrer les vidéos et les trier dans l'ordre des prédictions
    results = videos_df[videos_df["id"].isin(pred_ids)].copy()
    results['order'] = results['id'].apply(lambda x: list(pred_ids).index(x))
    results = results.sort_values('order')[["titre", "url"]]
    return results

# ===== Interface console =====
def main():
    print("\n🎥 Bienvenue dans le système de recommandation !\n")
    i = 1
    k= 5
    while True:
        question = input(f"Question n°{i} (ou 'exit') : ")
        if question.strip().lower() in ["exit", "quit", "stop"]:
            print("Fin du programme.")
            break

        results = recommander_video(question,top_k=k)
        print("\n📺 Recommandations :")
        for _, row in results.iterrows():
            print(f"- {row['titre']} ({row['url']})")

        i += 1
        print("\n------------------------------")

if __name__ == "__main__":
    main()
