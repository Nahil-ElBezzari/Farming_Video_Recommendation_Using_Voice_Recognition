from textaugment import EDA
import csv

augmentation = EDA()

input_file = "./Data/questions2.csv"

new_rows = []

with open(input_file, mode='r', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile, delimiter=';')
    header = next(reader)

    for row in reader:
        print(row)
        question, question_id = row[0].strip(), int(row[1].strip())
        new_question = augmentation.random_insertion(question)
        new_rows.append([new_question, question_id])
        new_question = augmentation.random_deletion(question)
        new_rows.append([new_question, question_id])

with open(input_file, mode='a', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, delimiter=';')
    writer.writerows(new_rows)

print(f"Fichier {input_file} mis à jour avec succès.")
