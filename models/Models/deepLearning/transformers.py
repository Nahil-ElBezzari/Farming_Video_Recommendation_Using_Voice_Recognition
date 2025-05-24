import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Charger les données
videos_df = pd.read_csv('Video2.csv')
training_df = pd.read_csv('trainingSet.csv')

# Prétraiter les titres et tags des vidéos pour obtenir une description complète de la vidéo
videos_df['description'] = videos_df['titre'] + " " + videos_df['tags'].apply(lambda x: " ".join(eval(x)) if x != 'set()' else "")

# Créer un vectoriseur TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')

# Appliquer le TF-IDF sur les titres/tags des vidéos et les questions
corpus_videos = videos_df['description'].tolist()
corpus_questions = training_df['question'].tolist()

# Combine les corpus pour transformer en une matrice TF-IDF
corpus = corpus_videos + corpus_questions
tfidf_matrix = vectorizer.fit_transform(corpus)

# Calculer les similarités cosinus entre les vidéos et les questions
cos_similarities = cosine_similarity(tfidf_matrix[:len(corpus_videos)], tfidf_matrix[len(corpus_videos):])

# Pour chaque question, trouver la vidéo la plus similaire
for i, question in enumerate(training_df['question']):
    best_video_index = cos_similarities[i].argmax()  # Trouver la vidéo avec la plus haute similarité
    print(f"Question: {question}")
    print(f"Meilleure vidéo: {videos_df.iloc[best_video_index]['titre']}")
    print(f"Similarité: {cos_similarities[i][best_video_index]}")
    print()
