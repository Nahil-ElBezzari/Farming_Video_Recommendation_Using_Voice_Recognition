import pandas as pd
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate

from tensorflow.keras import backend as K

from tensorflow.keras import Model


# Charger les fichiers CSV
videos_df = pd.read_csv("../../../Data/videos.csv", sep=";", encoding="utf-8") # Questions: 'question' column
questions_df = pd.read_csv("../../../Data/questions2.csv", sep=";", encoding="utf-8")  # Vidéos: 'title' & 'tags' columns

# 2. Préparer les questions et vidéos
questions = questions_df['question'].tolist()
titles = videos_df['titre'].tolist()
tags = videos_df['tags'].apply(lambda x: set(x.strip("[]").split(",") if isinstance(x, str) else [])).tolist()

# Concatenation des titres et tags pour chaque vidéo afin de créer une description complète
videos_descriptions = [f"{title} {' '.join(tags_list)}" for title, tags_list in zip(titles, tags)]

# Initialisation du modèle Sentence-BERT pour obtenir les embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions et vidéos en embeddings
questions_embeddings = model.encode(questions)
videos_embeddings = model.encode(videos_descriptions)

# 1. Créer des paires de données : (question, vidéo), où la sortie est 1 si la vidéo est pertinente, 0 sinon
# Pour simplifier, je vais générer des échantillons de manière aléatoire (tu peux affiner cette partie pour ton cas réel)

# Générer des paires de (question, vidéo) avec une étiquette (1 si la vidéo est pertinente, 0 sinon)
pairs = []
labels = []
for i in range(len(questions)):
    for j in range(len(videos_descriptions)):
        # Si la question et la vidéo sont "similaires", on attribue un label de 1 (pertinent)
        similarity_score = np.dot(questions_embeddings[i], videos_embeddings[j])
        label = 1 if similarity_score > 0.75 else 0  # Seuil arbitraire
        pairs.append((questions_embeddings[i], videos_embeddings[j]))
        labels.append(label)

# Convertir les paires en arrays numpy
pairs = np.array(pairs)
labels = np.array(labels)

# 2. Split des données en train et test
X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)


# 3. Créer l'architecture du réseau siamois
def euclidean_distance(vectors):
    x, y = vectors
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


# Créer l'encodeur pour les questions et vidéos
input_question = Input(shape=(questions_embeddings.shape[1],))
input_video = Input(shape=(videos_embeddings.shape[1],))

# Architecture partagée pour encoder les deux entrées
shared_dense = Dense(128, activation='relu')

encoded_question = shared_dense(input_question)
encoded_video = shared_dense(input_video)

# Calculer la distance euclidienne entre les deux embeddings
distance = Lambda(euclidean_distance)([encoded_question, encoded_video])

# Ajouter une couche de sortie pour la probabilité de pertinence (0 ou 1)
output = Dense(1, activation='sigmoid')(distance)

# Créer et compiler le modèle
siamese_model = Model(inputs=[input_question, input_video], outputs=output)
siamese_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Entraîner le modèle
siamese_model.fit([X_train[:, 0], X_train[:, 1]], y_train, epochs=5, batch_size=64,
                  validation_data=([X_test[:, 0], X_test[:, 1]], y_test))


# 5. Tester le modèle
def recommend_videos_with_siamese(question_embedding, top_k=10):
    # Calculer la distance entre la question et toutes les vidéos
    distances = []
    for video_embedding in videos_embeddings:
        pred = siamese_model.predict([np.array([question_embedding]), np.array([video_embedding])])
        distances.append(pred[0][0])

    # Trouver les top_k vidéos les plus proches
    top_indices = np.argsort(distances)[-top_k:]

    return [titles[i] for i in top_indices]


# Exemple de recommandation après l'entraînement
sample_question = "Comment gérer les semis en période de nut sécheresse ?"
question_embedding = model.encode([sample_question])[0]
recommended_videos = recommend_videos_with_siamese(question_embedding)
print("Vidéos recommandées:", recommended_videos)
