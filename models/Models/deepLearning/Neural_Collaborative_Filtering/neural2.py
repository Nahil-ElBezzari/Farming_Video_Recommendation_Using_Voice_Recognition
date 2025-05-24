import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from sklearn.preprocessing import LabelEncoder

# 1. Charger les fichiers CSV
videos_df = pd.read_csv("../../../Data/videos.csv", sep=";", encoding="utf-8") # Questions: 'question' column
questions_df = pd.read_csv("../../../Data/questions2.csv", sep=";", encoding="utf-8")  # Vidéos: 'title' & 'tags' columns

# 2. Préparer les questions et vidéos
questions = questions_df['question'].tolist()
titles = videos_df['titre'].tolist()
tags = videos_df['tags'].apply(lambda x: set(x.strip("[]").split(",") if isinstance(x, str) else [])).tolist()

# Concatenation des titres et tags pour chaque vidéo afin de créer une description complète
videos_descriptions = [f"{title} {' '.join(tags_list)}" for title, tags_list in zip(titles, tags)]

# 3. Charger le modèle Sentence-BERT pour obtenir les embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions et les vidéos en embeddings
questions_embeddings = model.encode(questions, convert_to_tensor=True)
videos_embeddings = model.encode(videos_descriptions, convert_to_tensor=True)

# 4. Utiliser FAISS pour accélérer la recherche des vidéos les plus similaires
# Créer un index FAISS à partir des embeddings des vidéos
dim = videos_embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dim)  # Utilise la distance euclidienne
faiss_index.add(np.array(videos_embeddings))  # Ajoute les embeddings des vidéos à l'index


# 5. Fonction de recommandation
def recommend_videos(question, top_k=10):
    # Encoder la question
    question_embedding = model.encode([question], convert_to_tensor=True)

    # Recherche des vidéos les plus proches
    D, I = faiss_index.search(np.array(question_embedding), top_k)  # D = distances, I = indices

    # Extraire les titres des vidéos recommandées
    recommended_videos = [titles[i] for i in I[0]]

    return recommended_videos


# Exemple de recommandation
sample_question = "Comment gérer les semis en période de nut sécheresse"
recommended_videos = recommend_videos(sample_question)
print("Vidéos recommandées:", recommended_videos)
