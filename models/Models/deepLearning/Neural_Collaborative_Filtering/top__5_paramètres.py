import pandas as pd

# Charger les données
df = pd.read_csv("hyperparameter_results_v2_1.csv")  # Remplace "fichier.csv" par le bon nom de fichier

# Trier les valeurs par accuracy décroissante et sélectionner les 5 meilleures
top_5 = df.nlargest(5, 'accuracy')

# Afficher les résultats
print(top_5)