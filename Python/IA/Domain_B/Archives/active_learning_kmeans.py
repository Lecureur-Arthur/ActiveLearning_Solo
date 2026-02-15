import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import random

# --- IMPORTATION POUR LE CLUSTERING ---
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"

model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

results_dir = script_dir / "AL_Results" / "Diversity_KMeans"
checkpoints_dir = results_dir / "checkpoints"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

MAX_X = 1920.0
MAX_Y = 1080.0

BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 5      
AL_LR = 1e-5

print(f"--- ACTIVE LEARNING (DIVERSITY - K-MEANS) ---")

if not model_A_path.exists():
    raise FileNotFoundError("Modèle A introuvable.")

# ==========================================
# 2. DATASET
# ==========================================
class ALDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        name = os.path.basename(path).rsplit('.', 1)[0]
        parts = name.split('_')
        try:
            x = float(parts[-2]) / MAX_X
            y = float(parts[-1]) / MAX_Y
        except:
            x, y = 0.0, 0.0
        target = torch.tensor([x, y], dtype=torch.float32)
        inputs = self.processor(image, return_tensors="pt")
        return inputs['pixel_values'].squeeze(0), target

def evaluate_mse(model, loader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for pix, tgt in loader:
            pix, tgt = pix.to(device), tgt.to(device)
            out = torch.sigmoid(model(pix).logits)
            loss = criterion(out, tgt)
            total_loss += loss.item()
    return total_loss / len(loader)

# ==========================================
# 3. EXTRACTION DE FEATURES
# ==========================================
def get_features(model, pool_files, processor):
    """
    Passe toutes les images dans le modèle et récupère le vecteur 
    avant la dernière couche (embedding).
    """
    print(f" -> Extraction des features sur {len(pool_files)} images...")
    model.eval()
    
    # On utilise un batch size un peu plus gros car pas de backprop
    loader = DataLoader(ALDataset(pool_files, processor), batch_size=64, shuffle=False)
    
    features_list = []
    
    with torch.no_grad():
        for pix, _ in tqdm(loader, desc="Features"):
            pix = pix.to(device)
            
            # Pour ResNet HuggingFace, on accède au backbone via .resnet
            outputs = model.resnet(pix)
            
            # pooler_output est de taille (Batch, 512, 1, 1)
            # On le transforme en (Batch, 512)
            feats = outputs.pooler_output.squeeze()
            features_list.append(feats.cpu().numpy())
            
    # On concatène tout en une seule grosse matrice numpy
    return np.concatenate(features_list, axis=0)

# ==========================================
# 4. SÉLECTION K-MEANS
# ==========================================
def select_samples_kmeans(model, pool_files, n_needed, processor):
    
    # 1. On récupère la "carte mentale" de toutes les images
    features = get_features(model, pool_files, processor)
    
    print(f" -> Clustering K-Means en {n_needed} groupes...")
    
    # 2. K-Means
    # On demande à l'algo de trouver 'n_needed' centres optimaux
    kmeans = KMeans(n_clusters=n_needed, random_state=42, n_init=10)
    kmeans.fit(features)
    
    # 3. Trouver les images les plus proches des centres (Centroids)
    # pairwise_distances_argmin_min retourne l'index du point le plus proche de chaque centre
    closest_indices, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, features)
    
    # Sélection (on s'assure qu'ils sont uniques, bien que K-Means le fasse généralement)
    unique_indices = np.unique(closest_indices)
    
    # Si jamais on a moins de points que prévu (cas rare de clusters vides), on complète
    selected_indices = list(unique_indices)
    while len(selected_indices) < n_needed:
        remaining_idx = [i for i in range(len(pool_files)) if i not in selected_indices]
        if not remaining_idx: break
        selected_indices.append(remaining_idx[0])

    selected_files = [pool_files[i] for i in selected_indices]
    
    # Mise à jour du pool
    remaining_pool = [pool_files[i] for i in range(len(pool_files)) if i not in selected_indices]
    
    return selected_files, remaining_pool

# ==========================================
# 5. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
if not files_B_pool: files_B_pool = list(path_train_B.glob("*.jpg"))

files_B_test = list(path_test_B.glob("**/*.jpg"))
if not files_B_test: files_B_test = list(path_test_B.glob("*.jpg"))

processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

labeled_B = []
results_mse = []
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS

# ==========================================
# 6. BASELINE (0%)
# ==========================================
print("\n--- BASELINE (0%) ---")
model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
initial_mse = evaluate_mse(model, test_loader)
results_mse.append(initial_mse)
print(f" -> MSE Initiale : {initial_mse:.6f}")

# ==========================================
# 7. BOUCLE ACTIVE LEARNING
# ==========================================
current_pool = files_B_pool

for pct in BUDGET_STEPS:
    print(f"\n>>> VISÉE : {pct}% du Domain B (Stratégie: Diversity K-Means)")
    
    target_total = int(len(files_B_pool) * (pct / 100.0))
    current_count = len(labeled_B)
    n_needed = target_total - current_count
    
    if n_needed > 0:
        # On utilise le modèle actuel pour extraire les features
        # (Car la représentation des images change au fur et à mesure qu'il apprend)
        new_data, current_pool = select_samples_kmeans(model, current_pool, n_needed, processor)
        
        labeled_B.extend(new_data)
        print(f" -> Ajout de {len(new_data)} images (Centroids).")
    
    # Entraînement
    mixed_files = files_A + labeled_B
    train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
    
    # Reset Modèle A
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=AL_LR)
    criterion = nn.MSELoss()
    
    for epoch in range(AL_EPOCHS):
        loop = tqdm(train_loader, desc=f"Train {pct}%", leave=False)
        for pix, tgt in loop:
            pix, tgt = pix.to(device), tgt.to(device)
            optimizer.zero_grad()
            out = torch.sigmoid(model(pix).logits)
            loss = criterion(out, tgt)
            loss.backward()
            optimizer.step()
            
    # Evaluation
    current_mse = evaluate_mse(model, test_loader)
    results_mse.append(current_mse)
    print(f" -> MSE Résultat : {current_mse:.6f}")
    
    # Sauvegarde
    folder_name = f"model_kmeans_{pct}percent"
    save_path = checkpoints_dir / folder_name
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

# ==========================================
# 8. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(BUDGET_WITH_ZERO, results_mse, 'd-', linewidth=2, color='green', label='Diversity (K-Means)')
plt.title("Active Learning Curve (Diversité)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

out_file = results_dir / "kmeans_curve.png"
plt.savefig(out_file)
print(f"\nTerminé ! Graphique : {out_file}")
plt.show()

print("\n--- RÉSUMÉ K-MEANS ---")
for p, m in zip(BUDGET_WITH_ZERO, results_mse):
    print(f"{p}% -> MSE: {m:.6f}")