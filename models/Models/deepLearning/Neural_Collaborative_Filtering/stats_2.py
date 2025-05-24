import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # Ou 'Agg'
# Charger les résultats du CSV
results_df = pd.read_csv("hyperparameter_results1.csv")

# Filtrer les résultats avec une précision > 95%
high_accuracy_df = results_df[results_df['accuracy'] > 0.95]

# Calculer la moyenne de chaque paramètre
mean_values = high_accuracy_df.mean(numeric_only=True)
print("Moyenne des paramètres pour accuracy > 95%:")
print(mean_values)

# Visualisation des distributions des paramètres
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Distribution des hyperparamètres pour accuracy > 95%", fontsize=14)

param_list = ['batch_size', 'hidden_dim', 'num_epochs', 'learning_rate', 'dropout', 'accuracy']

for ax, param in zip(axes.flat, param_list):
    ax.hist(high_accuracy_df[param], bins=10, edgecolor='black', alpha=0.7)
    ax.set_title(param)
    ax.set_xlabel(param)
    ax.set_ylabel("Fréquence")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
