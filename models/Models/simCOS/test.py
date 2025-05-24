# Installer les dépendances nécessaires avant d'exécuter le code
# Commandes pip :
# pip install sentence-transformers torch mysql-connector-python

from Data import createDB as db
import mysql.connector
from sentence_transformers import SentenceTransformer, util
import sys

def getTags(cursor, titre):
    tags_final = ""
    try:
        cursor.execute("""
            SELECT TAG FROM o_video
            JOIN ia_agriculture.t_video_tag tvt ON o_video.id = tvt.FK_O_VIDEO
            JOIN ia_agriculture.o_tag ot ON ot.ID_O_TAG = tvt.FK_O_TAG
            WHERE video_title = %s
        """, (titre,))
        result = cursor.fetchall()
        for (tag,) in result:
            tags_final += tag + " "  # Ajout d'un espace entre les tags

    except mysql.connector.Error as err:
        print(f"Error : {err}")
        sys.exit(2)

    return tags_final.strip()

def loadVideoData(connection):
    # Récupération des vidéos avec leurs tags
    data_videos = []
    try:
        cursor = connection.cursor()
        cursor.execute("SELECT video_title FROM o_video")
        videos = cursor.fetchall()

        for (video_title,) in videos:
            tags = getTags(cursor, video_title)
            data_videos.append({"titre": video_title, "tags": tags})

        return data_videos


    except Exception as e:
        print(f"Erreur lors de la récupération des vidéos : {e}")
    finally:
        cursor.close()
        connection.close()

def simCOS(connection, question,data_videos, top_n):
    # Chargement du modèle SBERT
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encodage de la question et des tags
    question_embedding = model.encode(question, convert_to_tensor=True)
    tags_embeddings = [model.encode(video['tags'], convert_to_tensor=True) for video in data_videos]

    # Calcul de la similarité cosinus
    scores = [util.pytorch_cos_sim(question_embedding, tag_emb)[0][0].item() for tag_emb in tags_embeddings]

    # Association des scores aux vidéos
    videos_scores = [
        {"titre": video["titre"], "tags": video["tags"], "score": score}
        for video, score in zip(data_videos, scores)
    ]

    # Tri des vidéos selon la similarité et sélection des top_n résultats
    videos_scores_sorted = sorted(videos_scores, key=lambda x: x["score"], reverse=True)[:top_n]

    # Affichage des résultats
    print(f"Top {top_n} recommandations pour la question :", question)
    for idx, video in enumerate(videos_scores_sorted, 1):
        print(f"{idx}. {video['titre']} (Score: {video['score']:.4f}) - Tags: {video['tags']}")

def main():
    connection = db.connect_to_database()

    data_videos = loadVideoData(connection)
    print("Video data loaded")

    while(True):
        question = input("\nQuestion: ")
        if(question == "0"):
            break
        simCOS(connection, question, data_videos,top_n=5)


if __name__ == "__main__":
    main()
