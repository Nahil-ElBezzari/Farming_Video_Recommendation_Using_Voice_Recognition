import torch

# Charger le modèle
model_path = "video_classifier_model_100.pth"  # Mets le bon chemin si besoin
model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# Afficher les noms des couches et leurs dimensions
for name, param in model_state_dict.items():
    print(f"{name}: {param.shape}")