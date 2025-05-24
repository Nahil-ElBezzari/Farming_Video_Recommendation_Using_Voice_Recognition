import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"



videos_df = pd.read_csv("../../../Data/videos.csv", sep=";", encoding="utf-8")
questions_df = pd.read_csv("../../../Data/questions2.csv", sep=";", encoding="utf-8")

# Initialiser le modèle Sentence-BERT pour encoder les questions
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions en embeddings
question_embeddings = sentence_model.encode(questions_df['question'].tolist())

# Encoder les labels (id des vidéos) avec LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(questions_df['id'].values)

# Diviser les données en train et test
X_train, X_test, y_train, y_test = train_test_split(question_embeddings, y, test_size=0.2, random_state=42)

# Convertir les données en tensors PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Dataset personnalisé pour PyTorch
class QuestionDataset(Dataset):
    def __init__(self, questions, labels):
        self.questions = questions
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]


# Créer les DataLoader pour le train et le test
train_dataset = QuestionDataset(X_train_tensor, y_train_tensor)
test_dataset = QuestionDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Définir le modèle de réseau de neurones
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VideoClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x


# Fonction pour sauvegarder le modèle
def save_model(model, filepath):
    torch.save(model.state_dict(), filepath)
    print(f"Modèle sauvegardé dans {filepath}")


# Fonction pour charger le modèle
def load_model(model, filepath):
    model.load_state_dict(torch.load(filepath))
    model.eval()  # Mettre le modèle en mode évaluation
    print(f"Modèle chargé depuis {filepath}")
    return model


# Entraînement du modèle
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")


# Évaluer le modèle
def evaluate_model(model, test_loader):
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())

    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%")


# Fonction pour recommander une vidéo en fonction de la question
def recommander_video(question, model, label_encoder, sentence_model, videos_df):
    # Obtenir l'embedding de la question en utilisant Sentence-BERT
    question_embedding = sentence_model.encode([question])  # Utiliser sentence_model ici
    question_tensor = torch.tensor(question_embedding, dtype=torch.float32)

    # Faire la prédiction avec le modèle de classification (réseau de neurones)
    model.eval()
    with torch.no_grad():
        output = model(question_tensor)
        _, predicted_class = torch.max(output, 1)

    # Convertir l'ID prédit en vidéo
    predicted_video_id = label_encoder.inverse_transform([predicted_class.item()])[0]

    # Trouver la vidéo correspondante
    video_info = videos_df[videos_df['id'] == predicted_video_id].iloc[0]

    return video_info[['titre', 'tags', 'url']]


# Fonction principale pour exécuter le programme
def main():
    # Initialiser le modèle
    input_dim = X_train.shape[1]  # Taille des embeddings
    hidden_dim = 128  # Nombre de neurones dans les couches cachées
    output_dim = len(label_encoder.classes_)  # Nombre de classes (vidéos)

    model = VideoClassifier(input_dim, hidden_dim, output_dim)

    # Définir la fonction de perte et l'optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Entraîner le modèle
    train_model(model, train_loader, criterion, optimizer, num_epochs=100)

    # Sauvegarder le modèle après l'entraînement
    save_model(model, "video_classifier_model_98-3p.pth")

    # Charger le modèle pour le test
    model = VideoClassifier(input_dim, hidden_dim, output_dim)  # Recréer l'architecture du modèle
    model = load_model(model, "video_classifier_model_98-3p.pth")

    # Évaluer le modèle sur le jeu de test
    evaluate_model(model, test_loader)


if __name__ == "__main__":
    main()
