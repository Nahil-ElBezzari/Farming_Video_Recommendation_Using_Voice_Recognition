import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

# Désactiver l'avertissement de symlinks sur Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hyperparamètres :
CONFIG = {
    'batch_size': 16,
    'hidden_dim': 128,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'validation_size': 0.1,
    'dropout': 0.3,
    'input_dim': None,
    'output_dim': None,
}

# Charger les fichiers CSV
videos_df = pd.read_csv("videos-question-form.csv", sep=";", encoding="utf-8")
questions_df = pd.read_csv("trainingSet.csv", sep=";", encoding="utf-8")

# Initialiser le modèle Sentence-BERT pour encoder les questions
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions en embeddings
question_embeddings = sentence_model.encode(questions_df['question'].tolist())

# Encoder les labels (id des vidéos)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(questions_df['id'].values)

# Split train / validation
X_train, X_validation, y_train, y_validation = train_test_split(
    question_embeddings, y, test_size=CONFIG['validation_size'], random_state=42
)

# Dimension des entrées
CONFIG['input_dim'] = X_train.shape[1]
CONFIG['output_dim'] = len(label_encoder.classes_)

# Convertir en tenseurs PyTorch
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)

# Dataset personnalisé
class QuestionDataset(Dataset):
    def __init__(self, questions, labels):
        self.questions = questions
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]

# DataLoaders
train_dataset = QuestionDataset(X_train_tensor, y_train_tensor)
validation_dataset = QuestionDataset(X_validation_tensor, y_validation_tensor)
train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

# Modèle de classification
class VideoClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_prob=CONFIG['dropout']):
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

#Sauvegarde modèle + config + labels
def save_model(model, config, filepath, label_encoder):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config,
        'labels': label_encoder.classes_.tolist()
    }
    torch.save(checkpoint, filepath)
    print(f"Modèle sauvegardé dans {filepath}")

# Chargement modèle
def load_model(model_class, filepath):
    checkpoint = torch.load(filepath)
    config = checkpoint['config']

    model = model_class(config['input_dim'], config['hidden_dim'], config['output_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(checkpoint['labels'])

    print(f"Modèle chargé depuis {filepath}")
    return model, config, label_encoder

# Entraînement
def train_model(model, train_loader, criterion, optimizer, num_epochs=CONFIG['num_epochs']):
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

# Évaluation
def evaluate_model(model, validation_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy : {accuracy * 100:.2f}%")

# Recommandation
def recommander_video(question, model, label_encoder, sentence_model, videos_df):
    question_embedding = sentence_model.encode([question])
    question_tensor = torch.tensor(question_embedding, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(question_tensor)
        probabilities = torch.softmax(output, dim=1)
        top_k_values, top_k_indices = torch.topk(probabilities, 4)

    predicted_video_ids = label_encoder.inverse_transform(top_k_indices.cpu().numpy().flatten())
    recommended_videos = videos_df[videos_df['id'].isin(predicted_video_ids)].copy()
    recommended_videos['order'] = recommended_videos['id'].apply(lambda x: list(predicted_video_ids).index(x))
    recommended_videos = recommended_videos.sort_values('order')[['titre', 'tags', 'url']]
    return recommended_videos

# Programme principal
def main():
    model = VideoClassifier(CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['output_dim'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    train_model(model, train_loader, criterion, optimizer, num_epochs=CONFIG['num_epochs'])
    save_model(model, CONFIG, "video_classifier_model.pth", label_encoder)

    # Test du modèle après rechargement
    model, loaded_config, loaded_label_encoder = load_model(VideoClassifier, "video_classifier_model.pth")
    evaluate_model(model, validation_loader)

if __name__ == "__main__":
    main()
