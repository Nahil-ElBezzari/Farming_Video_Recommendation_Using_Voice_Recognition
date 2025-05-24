import numpy as np, time, csv
from sentence_transformers import InputExample, losses, SentenceTransformer, SentenceTransformerTrainer
from torch.utils.data import DataLoader
from datasets import Dataset

#***DATASET CREATION***
csv_name = "extracted_sentences.csv"
corpus = []
with open("agri_ressources/"+csv_name, newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for line in csv_reader:
            corpus.extend(line)

start_time = time.perf_counter()
similarity_matrix = np.load("embedding/similarity_matrix.npy")
similarity_threshold = 0.75
opposite_threshold = 0.2
dict_dataset = {"sentence1":[], "sentence2":[], "score":[]}

for i in range(len(similarity_matrix)):
    for j in range(i + 1, len(similarity_matrix)):
        if (similarity_matrix[i, j] >= similarity_threshold) or (similarity_matrix[i, j] < opposite_threshold): #Only very similar or very different sentences
            dict_dataset["sentence1"].append(corpus[i])
            dict_dataset["sentence2"].append(corpus[j])
            dict_dataset["score"].append(similarity_matrix[i, j])
end_time = time.perf_counter()
run_time = end_time - start_time
print("✅ Sentences successfully paired !\nTime: "+str(run_time))


#***FINE TUNING***
start_time = time.perf_counter()

model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
dataset = Dataset.from_dict(dict_dataset)
train_loss = losses.CosineSimilarityLoss(model) #Loss function

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=dataset,
    loss=train_loss,
)
model.save_pretrained("embedding/models/sbert-fine-tuned-1005-v1")

end_time = time.perf_counter()
run_time = end_time - start_time
print("\n✅ Model successfuly trained !\nTime: "+str(run_time))