import torch
import pandas as pd
from neural1 import VideoClassifier, load_model, label_encoder, sentence_model
import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Paramètres
K = 4  # top-K vidéos suggérées
MODEL_PATH = "video_classifier_model.pth"

# Charger le modèle
input_dim = 384
hidden_dim = 128
output_dim = len(label_encoder.classes_)
model = VideoClassifier(input_dim, hidden_dim, output_dim)
model = load_model(model, MODEL_PATH)

# Charger le jeu de test
test_df = pd.read_csv("../../../Data/testSet.csv", sep=";", encoding="utf-8")

# Initialiser score total
mrr_total = 0.0
n_samples = len(test_df)

# Encoder les questions
questions = test_df["question"].tolist()
true_ids = test_df["id"].tolist()
question_embeddings = sentence_model.encode(questions)

for i, (embedding, true_id) in enumerate(zip(question_embeddings, true_ids)):
    question_tensor = torch.tensor([embedding], dtype=torch.float32)

    with torch.no_grad():
        output = model(question_tensor)
        probs = torch.softmax(output, dim=1)
        topk_probs, topk_indices = torch.topk(probs, K)

    predicted_ids = label_encoder.inverse_transform(topk_indices[0].cpu().numpy())

    # Calculer le Reciprocal Rank
    rr = 0.0
    for rank, pred_id in enumerate(predicted_ids, start=1):
        if pred_id == true_id:
            rr = 1.0 / rank
            break
    mrr_total += rr

# Moyenne
mean_mrr = mrr_total / n_samples
print(f"\nMean Reciprocal Rank (MRR) @ {K} : {mean_mrr:.4f}")
