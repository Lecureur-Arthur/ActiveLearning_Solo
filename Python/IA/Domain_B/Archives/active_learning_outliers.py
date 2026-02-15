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

# --- IMPORTATION POUR LES DISTANCES ---
from sklearn.metrics import pairwise_distances

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

results_dir = script_dir / "AL_Results" / "Diversity_Outliers"
checkpoints_dir = results_dir / "checkpoints"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

MAX_X = 1920.0
MAX_Y = 1080.0

BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 5      
AL_LR = 1e-5

print(f"--- ACTIVE LEARNING (DIVERSITY - OUTLIERS) ---")

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
def get_features(model, file_list, processor):
    """
    Extrait les vecteurs de caractéristiques pour une liste de fichiers.
    Retourne une matrice numpy (N_images, 512).
    """
    if not file_list:
        return np.array([])
        
    model.eval()
    loader = DataLoader(ALDataset(file_list, processor), batch_size=64, shuffle=False)
    features_list = []
    
    with torch.no_grad():
        for pix, _ in loader:
            pix = pix.to(device)
            outputs = model.resnet(pix)
            feats = outputs.pooler_output.squeeze()
            features_list.append(feats.cpu().numpy())
            
    if len(features_list) > 0:
        return np.concatenate(features_list, axis=0)
    return np.array([])

# ==========================================
# 4. SÉLECTION OUTLIERS (Farthest Point Sampling)
# ==========================================
def select_samples_outliers(model, pool_files, labeled_files, n_needed, processor):
    print(f" -> Calcul des distances (Farthest Point Sampling) pour {n_needed} images...")

    # 1. On extrait les features du POOL (Candidats)
    pool_features = get_features(model, pool_files, processor)
    
    # 2. On extrait les features de ce qu'on a DÉJÀ (Labeled Set + Domain A)
    # Important : On veut être loin de TOUT ce qu'on connait déjà
    labeled_features = get_features(model, labeled_files, processor)
    
    # 3. Calcul de la distance initiale
    # Pour chaque point du Pool, on trouve la distance min vers le set Labeled
    # distance_matrix shape : (N_Pool, N_Labeled)
    dists = pairwise_distances(pool_features, labeled_features, metric='euclidean')
    
    # min_dists shape : (N_Pool,) -> La distance au voisin le plus proche dans le set connu
    min_dists = np.min(dists, axis=1)
    
    selected_indices = []
    
    # 4. Boucle de sélection itérative (Greedy)
    # On ajoute les points un par un pour mettre à jour les distances dynamiquement
    for _ in range(n_needed):
        # A. On prend l'image qui a la plus grande distance minimale (la plus isolée)
        idx_farthest = np.argmax(min_dists)
        selected_indices.append(idx_farthest)
        
        # B. Mise à jour des distances
        # Maintenant que ce point est "connu", les autres points du pool ne doivent pas lui ressembler
        new_labeled_feat = pool_features[idx_farthest].reshape(1, -1)
        
        # Distances entre tout le pool et ce nouveau point choisi
        dists_to_new = pairwise_distances(pool_features, new_labeled_feat, metric='euclidean').flatten()
        
        # On met à jour min_dists : on garde le minimum entre l'ancienne distance et la nouvelle
        min_dists = np.minimum(min_dists, dists_to_new)
        
        # On met la distance du point choisi à -1 pour ne pas le re-sélectionner
        min_dists[idx_farthest] = -1.0

    selected_files = [pool_files[i] for i in selected_indices]
    
    # Mise à jour du pool restant
    remaining_pool = [pool_files[i] for i in range(len(pool_files)) if i not in selected_indices]
    
    return selected_files, remaining_pool

# ==========================================
# 5. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg")) # Le Domain A est notre "Base de connaissance"
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
    print(f"\n>>> VISÉE : {pct}% du Domain B (Stratégie: Outliers)")
    
    target_total = int(len(files_B_pool) * (pct / 100.0))
    current_count = len(labeled_B)
    n_needed = target_total - current_count
    
    if n_needed > 0:
        # L'ensemble "Déjà connu" (Labeled) c'est : Les fichiers de A + Les fichiers de B déjà choisis
        known_files = files_A + labeled_B
        
        # Sélection
        new_data, current_pool = select_samples_outliers(model, current_pool, known_files, n_needed, processor)
        
        labeled_B.extend(new_data)
        print(f" -> Ajout de {len(new_data)} images (Outliers).")
    
    # Entraînement
    mixed_files = files_A + labeled_B
    train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
    
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
    folder_name = f"model_outliers_{pct}percent"
    save_path = checkpoints_dir / folder_name
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

# ==========================================
# 8. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(BUDGET_WITH_ZERO, results_mse, 'X-', linewidth=2, color='darkred', label='Diversity (Outliers)')
plt.title("Active Learning Curve (Outliers/Exploration)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

out_file = results_dir / "outliers_curve.png"
plt.savefig(out_file)
print(f"\nTerminé ! Graphique : {out_file}")
plt.show()

print("\n--- RÉSUMÉ OUTLIERS ---")
for p, m in zip(BUDGET_WITH_ZERO, results_mse):
    print(f"{p}% -> MSE: {m:.6f}")