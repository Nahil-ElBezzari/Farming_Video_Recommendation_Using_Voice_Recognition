"""
Avant de lancer le script il faut faire cette commande
nltk.download('stopwords')
pip install google-api-python-client
pip install mysql-connector-python

"""

import googleapiclient.discovery
import mysql.connector
import sys
import re
from nltk.corpus import stopwords

def parse_config_file(filename):
    data = {}
    with open(filename, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split(" : ")
            if len(parts) == 2:
                key, value = parts
                data[key.strip()] = value.strip()
    return data


config = parse_config_file("config.txt")
API_KEY = config.get("API_KEY")
PLAYLIST_ID = config.get("PLAYLIST_ID")
DB_HOST = config.get("DB_HOST")
DB_USER = config.get("DB_USER")
DB_PASSWORD = config.get("DB_PASSWORD")
DB_NAME = config.get("DB_NAME")


def create_youtube_service():
    """Initialise l'API YouTube."""
    try:
        return googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    except Exception as e:
        print(f"Erreur lors de la création du service YouTube : {str(e)}")
        sys.exit(1)

def connect_to_database():
    """Connecte à la base de données MySQL."""
    try:
        return mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            auth_plugin="mysql_native_password",
            charset="utf8mb4"
        )
    except mysql.connector.Error as err:
        print(f"Erreur de connexion à la base de données : {err}")
        sys.exit(2)

def insert_video_and_tags(connection, video_title, url, tags):
    """Insère une vidéo et ses tags dans la base de données."""
    try:
        cursor = connection.cursor()

        # Vérifier si la vidéo existe déjà
        cursor.execute("SELECT id FROM o_video WHERE video_title = %s", (video_title,))
        result = cursor.fetchone()

        # Ajout pour consommer le résultat restant
        cursor.fetchall()

        if result:
            video_id = result[0]
            #print(f"Vidéo '{video_title}' existe déjà avec l'ID {video_id}.")
        else:
            cursor.execute("INSERT INTO o_video (video_title, url) VALUES (%s, %s)", (video_title, url))
            video_id = cursor.lastrowid
            #print(f"Vidéo '{video_title}' insérée avec succès avec l'ID {video_id}.")

        # Insérer les tags associés à la vidéo
        for tag in tags:
            if tag.lower() == "none":
                continue

            # Vérifier si le tag existe déjà
            cursor.execute("SELECT ID_O_TAG FROM o_tag WHERE TAG = %s", (tag,))
            result = cursor.fetchone()

            # Ajout pour consommer le résultat restant
            cursor.fetchall()

            if result:
                tag_id = result[0]
            else:
                cursor.execute("INSERT INTO o_tag (TAG) VALUES (%s)", (tag,))
                tag_id = cursor.lastrowid

            # Vérifier si la relation existe déjà
            cursor.execute("SELECT * FROM t_video_tag WHERE FK_O_TAG = %s AND FK_O_VIDEO = %s", (tag_id, video_id))
            relation = cursor.fetchone()

            # Ajout pour consommer le résultat restant
            cursor.fetchall()

            if not relation:
                cursor.execute("INSERT INTO t_video_tag (FK_O_TAG, FK_O_VIDEO) VALUES (%s, %s)", (tag_id, video_id))
                #print(f"Relation insérée : Vidéo '{video_title}' -> Tag '{tag}'.")

        connection.commit()
    except mysql.connector.Error as err:
        print(f"Erreur lors de l'insertion dans la base de données : {err}")
    finally:
        cursor.close()

def get_videos_from_playlist(youtube, playlist_id):
    """Récupère les vidéos d'une playlist YouTube."""
    stop_words_fr = set(stopwords.words("french"))
    next_page_token = None
    video_data = []

    while True:
        try:
            playlist_request = youtube.playlistItems().list(
                playlistId=playlist_id,
                part="snippet",
                maxResults=50,
                pageToken=next_page_token
            )
            playlist_response = playlist_request.execute()

            for item in playlist_response['items']:
                video_title = item['snippet']['title']
                video_id = item['snippet']['resourceId']['videoId']
                url = f"https://www.youtube.com/watch?v={video_id}"

                video_request = youtube.videos().list(
                    part="snippet",
                    id=video_id
                )
                video_response = video_request.execute()

                tags = video_response['items'][0]['snippet'].get('tags', [])
                tags = tokenize_tags(tags)
                #enlève les stop word comme "le","la"
                tags ={tag for tag in tags if tag.lower() not in stop_words_fr}

                video_data.append((video_title, tags, url))

            next_page_token = playlist_response.get('nextPageToken')
            if not next_page_token:
                break

        except googleapiclient.errors.HttpError as err:
            print(f"Erreur HTTP lors de l'appel à l'API YouTube : {err}")
            sys.exit(1)
        except Exception as e:
            print(f"Erreur inattendue : {str(e)}")
            sys.exit(1)

    return video_data

def tokenize_tags(tags):
    tokens = set()
    for phrase in tags:
        phrase = phrase.strip()
        words = re.findall(r"\b\w+['’]\w+\b|\b\w+\b", phrase.lower())
        for word in words:
            if "'" in word or "’" in word:
                parts = re.split(r"['’]", word)
                tokens.update(parts)
            else:
                tokens.add(word)
    return tokens

def main():
    youtube = create_youtube_service()
    connection = connect_to_database()

    print(" Récupération des vidéos de la playlist et insertion dans la base de données...\n")
    video_data = get_videos_from_playlist(youtube, PLAYLIST_ID)

    for video_title, tags, url in video_data:
        insert_video_and_tags(connection, video_title, url, tags)

    connection.close()
    print("\n✅ Toutes les vidéos ont été insérées avec succès !")


if __name__ == "__main__":
    main()