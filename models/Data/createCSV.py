import createDB as DB
import csv
import googleapiclient.discovery
import sys
from nltk.corpus import stopwords
import time
import re

# Charger la configuration à partir du fichier
config = DB.parse_config_file("config.txt")
API_KEY = config.get("API_KEY")
PLAYLIST_ID = config.get("PLAYLIST_ID")
######################################################################
#############TRAITEMENTS##############################################
######################################################################
def remove_special_characters(text):
    """Supprime les caractères spéciaux d'une chaîne de texte, en conservant les lettres, chiffres, espaces, caractères accentués, apostrophes et tirets."""
    # Conserve les lettres, chiffres, espaces, caractères accentués, apostrophes et tirets
    return re.sub(r'[^a-zA-Z0-9\sàáâäãåçèéêëìíîïòóôöõùúûüÿýœæÆÇÈÉÊËÌÍÎÏÒÓÔÖÕÙÚÛÜŸÝŒ\'-]', '', text)
def clean_titles_in_csv(filename):
    """Parcourt un fichier CSV et supprime les caractères spéciaux des titres."""
    cleaned_data = []

    try:
        with open(filename, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                if row:  # Vérifie que la ligne n'est pas vide
                    # Supprime les caractères spéciaux du titre (première colonne)
                    cleaned_row = [remove_special_characters(row[0])] + row[1:]
                    cleaned_data.append(cleaned_row)

        # Écrit les données nettoyées dans le fichier CSV
        with open(filename, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(cleaned_data)

    except FileNotFoundError:
        print(f"Le fichier {filename} n'existe pas.")
    except Exception as e:
        print(f"Erreur lors de la lecture ou de l'écriture du fichier : {str(e)}")
def remove_emojis(text):
    """Supprime les emojis d'une chaîne de texte."""
    emoji_pattern = re.compile(
        "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    return emoji_pattern.sub(r'', text)
def clean_csv():
    """Parcourt un fichier CSV et supprime les emojis de chaque cellule."""
    filename = '/Data/videos.csv'
    cleaned_data = []

    try:
        with open(filename, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.reader(file, delimiter=';')
            for row in reader:
                cleaned_row = [remove_emojis(cell) for cell in row]
                cleaned_data.append(cleaned_row)

        # Écrit les données nettoyées dans le fichier CSV
        with open(filename, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.writer(file, delimiter=';')
            writer.writerows(cleaned_data)

    except FileNotFoundError:
        print(f"Le fichier {filename} n'existe pas.")
    except Exception as e:
        print(f"Erreur lors de la lecture ou de l'écriture du fichier : {str(e)}")
######################################################################
######################################################################
######################################################################
def create_youtube_service():
    """Initialise l'API YouTube."""
    try:
        return googleapiclient.discovery.build("youtube", "v3", developerKey=API_KEY)
    except Exception as e:
        print(f"Erreur lors de la création du service YouTube : {str(e)}")
        sys.exit(1)

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

                # Extraire et traiter les tags
                tags = video_response['items'][0]['snippet'].get('tags', [])
                tags = DB.tokenize_tags(tags)  # Assurez-vous que cette fonction existe
                tags = {tag for tag in tags if tag.lower() not in stop_words_fr}

                # Ajouter les données vidéo dans une liste
                video_data.append([video_title, tags, url])  # Utiliser une liste

            # Si une page suivante existe, la traiter
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

def emptyCSV():
    """Vide le fichier CSV."""
    filename = 'videos.csv'
    with open(filename, 'w', newline='', encoding='utf-8'):
        pass

def insertCSV(videoData):
    """Insère les données vidéo dans le fichier CSV."""
    id = 1
    filename = 'videos.csv'
    with open(filename, mode='a', newline='', encoding='utf-8') as file:  # Ajout de l'encodage 'utf-8'
        writer = csv.writer(file, delimiter=';')
        for row in videoData:
            row.append(id)  # Ajouter l'ID à la ligne
            writer.writerow(row)  # Écrire la ligne dans le CSV
            id += 1

def main():
    print("Traitement en cours ...")
    start_time = time.time()

    time_yt_deb = time.time()
    # Créer le service YouTube
    youtube = create_youtube_service()
    # Récupérer les vidéos de la playlist YouTube
    videoData = get_videos_from_playlist(youtube, PLAYLIST_ID)
    print(f"le temps API : {time.time() - time_yt_deb:.2f}")

    time_inser_deb = time.time()
    # Vide le CSV existant
    emptyCSV()
    # Remplir le CSV avec les vidéos récupérées de YouTube
    insertCSV(videoData)
    print(f"le temps d'insertion : {time.time() - time_inser_deb:.2f}")

    # traitement sur les données
    time_trait_deb = time.time()
    clean_csv()
    filename = 'videos.csv'
    clean_titles_in_csv(filename)
    print(f"Le temps de traitement : {time.time() - time_trait_deb:.2f} ")

    print(f"Le temps de traitement et d'insertion total est de {time.time() - start_time:.2f} secondes.")
if __name__ == "__main__":
    main()
