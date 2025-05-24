import time, csv
import numpy as np
start_time = time.perf_counter()

matrix_name = "similarity_matrix.npy"
csv_name = "extracted_sentences.csv"

# Extracting sentences from the csv file for fine-tuning
corpus = []
with open("agri_ressources/"+csv_name, newline='', encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter=';')
        for line in csv_reader:
            corpus.extend(line)

# Creating similarities between sentences
from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
embeddings = model.encode(corpus, convert_to_tensor=True)
similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings).numpy()
np.save("embedding/"+matrix_name, similarity_matrix)

end_time = time.perf_counter()
run_time = end_time - start_time
print("\n✅ Similarity matrix successfully saved to: "+matrix_name+"!\nTime: "+str(run_time))