import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score


# Hyperparamètres
CONFIG = {
    'batch_size': 512,
    'hidden_dim': 500,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'validation_size': 0.1,
    'dropout': 0.5,
    'input_dim': None,
}

# Charger les fichiers CSV
videos_df = pd.read_csv("../../../Data/videos-question-form.csv", sep=";", encoding="utf-8")
questions_df = pd.read_csv("../../../Data/trainingSet.csv", sep=";", encoding="utf-8")

# Initialiser Sentence-BERT
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Encoder les questions en embeddings
question_embeddings = sentence_model.encode(questions_df['question'].tolist())

# Encoder les labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(questions_df['id'].values)

# Diviser les données
X_train, X_validation, y_train, y_validation = train_test_split(question_embeddings, y, test_size=CONFIG['validation_size'], random_state=42)
CONFIG['input_dim'] = X_train.shape[1]

# Convertir en tensors
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

# DataLoader
train_loader = DataLoader(QuestionDataset(X_train_tensor, y_train_tensor), batch_size=CONFIG['batch_size'], shuffle=True)
validation_loader = DataLoader(QuestionDataset(X_validation_tensor, y_validation_tensor), batch_size=CONFIG['batch_size'], shuffle=False)

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

# Entraînement du modèle
def train_model(model, train_loader, criterion, optimizer, num_epochs):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Évaluation du modèle
def evaluate_model(model, validation_loader):
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for inputs, labels in validation_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.numpy())
            y_true.extend(labels.numpy())
    print(f"Accuracy: {accuracy_score(y_true, y_pred) * 100:.2f}%")

# Fonction principale
def main():
    model = VideoClassifier(CONFIG['input_dim'], CONFIG['hidden_dim'], len(label_encoder.classes_), CONFIG['dropout'])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    train_model(model, train_loader, criterion, optimizer, CONFIG['num_epochs'])
    evaluate_model(model, validation_loader)

if __name__ == "__main__":
    main()