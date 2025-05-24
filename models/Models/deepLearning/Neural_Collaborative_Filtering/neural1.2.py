import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import os

# Éviter l'avertissement sur Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Hyperparamètres fixés pour correspondre au modèle sauvegardé
CONFIG = {
    'batch_size': 16,
    'hidden_dim': 256,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'validation_size': 0.1,
    'dropout': 0.5,
    'input_dim': 384,
    'output_dim': 47
}

# Charger les données
videos_df = pd.read_csv("../../../Data/videos-question-form.csv", sep=";", encoding="utf-8")
questions_df = pd.read_csv("../../../Data/trainingSet.csv", sep=";", encoding="utf-8")

# Sentence-BERT
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions
question_embeddings = sentence_model.encode(questions_df['question'].tolist())

# Encoder les labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(questions_df['id'].values)

# Train/val split
X_train, X_validation, y_train, y_validation = train_test_split(
    question_embeddings, y, test_size=CONFIG['validation_size'], random_state=42
)

# Convertir en tenseurs
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_validation_tensor = torch.tensor(y_validation, dtype=torch.long)

# Dataset PyTorch
class QuestionDataset(Dataset):
    def __init__(self, questions, labels):
        self.questions = questions
        self.labels = labels

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        return self.questions[idx], self.labels[idx]

train_dataset = QuestionDataset(X_train_tensor, y_train_tensor)
validation_dataset = QuestionDataset(X_validation_tensor, y_validation_tensor)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=CONFIG['batch_size'], shuffle=True)

# Réseau de neurones correspondant exactement au modèle sauvegardé
class VideoClassifier(nn.Module):
    def __init__(self, input_dim=384, hidden_dim=128, output_dim=47, dropout_prob=0.3):
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

# Sauvegarde
def save_model(model, config, filepath):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'config': config
    }
    torch.save(checkpoint, filepath)
    print(f" Modèle sauvegardé dans {filepath}")

# Chargement
def load_model(model_class, filepath):
    checkpoint = torch.load(filepath, map_location=torch.device('cpu'))
    config = checkpoint['config']
    model = model_class(config['input_dim'], config['hidden_dim'], config['output_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f" Modèle chargé depuis {filepath}")
    return model, config

# Entraînement
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f" Epoch {epoch + 1}/{num_epochs} - Loss: {total_loss / len(train_loader):.4f}")

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
    acc = accuracy_score(y_true, y_pred)
    print(f"\n Accuracy sur validation : {acc * 100:.2f}%")

# Recommandation
def recommander_video(question, model, label_encoder, sentence_model, videos_df, k=4):
    question_embedding = sentence_model.encode([question])
    question_tensor = torch.tensor(question_embedding, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        output = model(question_tensor)
        probs = torch.softmax(output, dim=1)
        topk_vals, topk_idx = torch.topk(probs, k)

    predicted_ids = label_encoder.inverse_transform(topk_idx[0].cpu().numpy())
    recommended = videos_df[videos_df['id'].isin(predicted_ids)][['titre', 'tags', 'url']]
    return recommended

# Main
def main():
    model = VideoClassifier(CONFIG['input_dim'], CONFIG['hidden_dim'], CONFIG['output_dim'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])

    # Entraînement
    train_model(model, train_loader, criterion, optimizer, CONFIG['num_epochs'])

    # Sauvegarde
    save_model(model, CONFIG, "video_classifier_model.pth")

    # Rechargement + Évaluation
    model, loaded_config = load_model(VideoClassifier, "video_classifier_model.pth")
    evaluate_model(model, validation_loader)

if __name__ == "__main__":
    main()
