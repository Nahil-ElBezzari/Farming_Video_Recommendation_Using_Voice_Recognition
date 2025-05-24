import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from itertools import product
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os
import time

# Désactiver l'avertissement de symlinks sur Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
time_deb = time.time()
# Charger les fichiers CSV
videos_df = pd.read_csv("../../../Data/videos-question-form.csv", sep=";", encoding="utf-8")
questions_df = pd.read_csv("../../../Data/trainingSet.csv", sep=";", encoding="utf-8")

# Initialiser le modèle Sentence-BERT pour encoder les questions
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions en embeddings
question_embeddings = sentence_model.encode(questions_df['question'].tolist())

# Encoder les labels (id des vidéos) avec LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(questions_df['id'].values)

# Diviser les données en train et validation
X_train, X_validation, y_train, y_validation = train_test_split(question_embeddings, y, test_size=0.1, random_state=42)

# Convertir les données en tensors PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)


# Dataset personnalisé pour PyTorch
class QuestionDataset(Dataset):
    def __init__(self, questions, labels):
        self.questions = questions
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]


# Créer les datasets
train_dataset = QuestionDataset(X_train_tensor, y_train_tensor)
validation_dataset = QuestionDataset(X_validation_tensor, y_validation_tensor)

# Définition de la grille d'hyperparamètres
param_grid = {
    'batch_size': [512],
    'hidden_dim': [600, 700, 800],
    'num_epochs': [75],
    'learning_rate': [0.001, 0.0009, 0.0008, 0.0007, 0.0006, 0.0005],
    'dropout': [0.4, 0.5, 0.6]
}


# Définition du modèle
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob):
        super(VideoClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


# Liste pour stocker les résultats
results = []

# Tester toutes les combinaisons possibles
for batch_size, hidden_dim, num_epochs, learning_rate, dropout in product(*param_grid.values()):

    # Définir la configuration actuelle
    CONFIG = {
        'batch_size': batch_size,
        'hidden_dim': hidden_dim,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'input_dim': X_train.shape[1]  # Taille des embeddings
    }

    # Créer les DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    # Initialiser le modèle
    model = VideoClassifier(CONFIG['input_dim'], hidden_dim, len(label_encoder.classes_), dropout_prob=dropout)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Entraînement du modèle
    for epoch in range(num_epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Évaluation du modèle
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)

    # Enregistrer les résultats
    results.append([batch_size, hidden_dim, num_epochs, learning_rate, dropout, accuracy])
    print(f"Testé: {CONFIG} -> Accuracy: {accuracy:.4f}")

# Sauvegarder les résultats dans un CSV
results_df = pd.DataFrame(results,
                          columns=['batch_size', 'hidden_dim', 'num_epochs', 'learning_rate', 'dropout', 'accuracy'])
results_df.to_csv("hyperparameter_results.csv", index=False)
print("Résultats enregistrés dans hyperparameter_results.csv")
print(f" Temps total du programme : {time.time() - time_deb} ")
