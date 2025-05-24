import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Agg'


import seaborn as sns

# Charger les résultats depuis le fichier CSV
results_df = pd.read_csv("hyperparameter_results.csv")

# Afficher les premières lignes pour vérifier le contenu
print(results_df.head())

# Calculer la moyenne de l'accuracy pour chaque combinaison de paramètres
mean_accuracy_per_batch_size = results_df.groupby('batch_size')['accuracy'].mean()
mean_accuracy_per_hidden_dim = results_df.groupby('hidden_dim')['accuracy'].mean()
mean_accuracy_per_num_epochs = results_df.groupby('num_epochs')['accuracy'].mean()
mean_accuracy_per_learning_rate = results_df.groupby('learning_rate')['accuracy'].mean()
mean_accuracy_per_dropout = results_df.groupby('dropout')['accuracy'].mean()

# Afficher les moyennes
print("\nMoyenne de l'accuracy pour chaque batch_size :")
print(mean_accuracy_per_batch_size)

print("\nMoyenne de l'accuracy pour chaque hidden_dim :")
print(mean_accuracy_per_hidden_dim)

print("\nMoyenne de l'accuracy pour chaque num_epochs :")
print(mean_accuracy_per_num_epochs)

print("\nMoyenne de l'accuracy pour chaque learning_rate :")
print(mean_accuracy_per_learning_rate)

print("\nMoyenne de l'accuracy pour chaque dropout :")
print(mean_accuracy_per_dropout)

# Création des graphiques

# 1. Graphique des moyennes d'accuracy en fonction du batch_size
plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_accuracy_per_batch_size.index, y=mean_accuracy_per_batch_size.values, marker='o')
plt.title('Impact de batch_size sur l\'accuracy')
plt.xlabel('batch_size')
plt.ylabel('Moyenne de l\'accuracy')
plt.grid(True)
plt.show()

# 2. Graphique des moyennes d'accuracy en fonction du hidden_dim
plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_accuracy_per_hidden_dim.index, y=mean_accuracy_per_hidden_dim.values, marker='o')
plt.title('Impact de hidden_dim sur l\'accuracy')
plt.xlabel('hidden_dim')
plt.ylabel('Moyenne de l\'accuracy')
plt.grid(True)
plt.show()

# 3. Graphique des moyennes d'accuracy en fonction du num_epochs
plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_accuracy_per_num_epochs.index, y=mean_accuracy_per_num_epochs.values, marker='o')
plt.title('Impact de num_epochs sur l\'accuracy')
plt.xlabel('num_epochs')
plt.ylabel('Moyenne de l\'accuracy')
plt.grid(True)
plt.show()

# 4. Graphique des moyennes d'accuracy en fonction du learning_rate
plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_accuracy_per_learning_rate.index, y=mean_accuracy_per_learning_rate.values, marker='o')
plt.title('Impact de learning_rate sur l\'accuracy')
plt.xlabel('learning_rate')
plt.ylabel('Moyenne de l\'accuracy')
plt.grid(True)
plt.show()

# 5. Graphique des moyennes d'accuracy en fonction du dropout
plt.figure(figsize=(10, 6))
sns.lineplot(x=mean_accuracy_per_dropout.index, y=mean_accuracy_per_dropout.values, marker='o')
plt.title('Impact de dropout sur l\'accuracy')
plt.xlabel('dropout')
plt.ylabel('Moyenne de l\'accuracy')
plt.grid(True)
plt.show()

# 6. Heatmap pour observer les relations entre batch_size, hidden_dim, et accuracy
pivot_df = results_df.pivot_table(index='batch_size', columns='hidden_dim', values='accuracy', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_df, annot=True, cmap='Blues', fmt='.2f')
plt.title('Heatmap de l\'accuracy selon batch_size et hidden_dim')
plt.xlabel('hidden_dim')
plt.ylabel('batch_size')
plt.show()

# 7. Graphique combiné pour observer les relations entre plusieurs paramètres
sns.pairplot(results_df[['batch_size', 'hidden_dim', 'num_epochs', 'learning_rate', 'dropout', 'accuracy']], hue='accuracy')
plt.suptitle('Pairplot des hyperparamètres et de l\'accuracy', y=1.02)
plt.show()
