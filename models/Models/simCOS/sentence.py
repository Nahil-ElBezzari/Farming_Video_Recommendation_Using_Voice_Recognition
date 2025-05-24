import csv, time, torch

#__________DATA EXTRACTION__________
questions_file = "./Data/question.csv"
video_file = "./Data/videos-question-form.csv"
def extract_questions(file_path):
    questions_dict = {}
    
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file,delimiter=";")
        for row in reader:
            if len(row) == 2: 
                question, number = row[0].strip(), int(row[1].strip())
                questions_dict[question] = number
    
    return questions_dict
def extract_videos(file_path):
    videos_dict = {}

    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file,delimiter=";")
        for row in reader:
            title, tags, link, number = row[0].strip(), row[1].strip(), row[2].strip(), int(row[3].strip())
            videos_dict[title] = number
    
    return videos_dict

questions = extract_questions(questions_file)
videos = extract_videos(video_file)

#__________SIMILARITY__________
start_time = time.perf_counter()
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-mpnet-base-v2")
end_time = time.perf_counter()
run_time = end_time - start_time
print("\nTemps d'importation du modèle SBERT: "+str(run_time))

start_time = time.perf_counter()
videos_embedding = model.encode(list(videos.keys()), convert_to_tensor=True)
end_time = time.perf_counter()
run_time = end_time - start_time
print("\nTemps de vectorisation des titres des vidéos: "+str(run_time)+"\nTemps moyen: "+str(run_time/len(list(videos.keys()))))

mapping = {}
start_time = time.perf_counter()
for query in list(questions.keys()):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarity_scores = model.similarity(query_embedding, videos_embedding)[0]
    score, idx = torch.topk(similarity_scores, k=1)
    best_video = list(videos.keys())[idx]
    mapping[query]=best_video
end_time = time.perf_counter()
run_time = end_time - start_time
print("\nTemps de vectorisation des questions et de calcul de correspondance: "+str(run_time)+"\nTemps moyen: "+str(run_time/len(list(questions.keys()))))

#__________VERIFICATION__________
total_score = 0
for input,output in mapping.items():
    if(questions[input]==videos[output]):
        total_score+=1
print("\nAccuracy: "+str(total_score/len(list(questions.keys()))*100) + "%")
