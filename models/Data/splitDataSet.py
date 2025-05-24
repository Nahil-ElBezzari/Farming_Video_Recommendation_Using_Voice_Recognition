
import time

t_deb = time.time()
import pandas as pd
import random

df = pd.read_csv("questions2.csv", sep=";", on_bad_lines='skip')

df = df.dropna(subset=['id'])

df['id'] = df['id'].astype(int)

print(df.head())


grouped = df.groupby('id')


training_set = []
validation_set = []
test_set = []

# Parcourir chaque groupe d'ID
for id, group in grouped:
    # Mélanger les questions de chaque groupe de manière aléatoire
    questions = group.sample(frac=1, random_state=42)  # random_state pour reproductibilité

    # Vérifier s'il y a au moins 50 questions pour un ID donné
    if len(questions) >= 150:
        # Séparer en trois ensembles
        training_set.append(questions.head(135))
        #validation_set.append(questions.iloc[135:142])
        test_set.append(questions.iloc[135:150])

# Concatenation des sous-ensembles pour chaque ensemble complet
training_df = pd.concat(training_set)
#validation_df = pd.concat(validation_set)
test_df = pd.concat(test_set)

# Sauvegarder les ensembles dans des fichiers CSV avec séparateur ;
training_df.to_csv("trainingSet.csv", sep=";", index=False)
#validation_df.to_csv("validationSet.csv", sep=";", index=False)
test_df.to_csv("testSet.csv", sep=";", index=False)

print("Les fichiers trainingSet.csv, validationSet.csv, et testSet.csv ont été créés avec le séparateur ;.")

print(f"Temps de split = {time.time() - t_deb:.3f}")