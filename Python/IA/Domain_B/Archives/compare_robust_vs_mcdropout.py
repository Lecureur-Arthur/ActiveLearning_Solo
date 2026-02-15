import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt
import random
import numpy as np
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

# Chemins
path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Chemin vers les checkpoints existants du MC Dropout
path_mcdrop_checkpoints = script_dir / "AL_Results" / "Uncertainty_MC_Dropout" / "checkpoints"

# Dossier de sortie
output_dir = script_dir / "AL_Results" / "Comparison_Robust"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
MAX_X = 1920.0
MAX_Y = 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
AL_EPOCHS = 5      
AL_LR = 1e-5 
N_ROUNDS_RANDOM = 3  # Nombre de fois qu'on répète le Random pour avoir les barres d'erreur

print("--- COMPARATEUR ROBUSTE (RANDOM vs MC DROPOUT) ---")
print(f"Modèles MC Dropout : {path_mcdrop_checkpoints}")

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
        pixel_values = inputs['pixel_values'].squeeze(0)
        return pixel_values, target

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
# 3. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
if not files_B_pool: files_B_pool = list(path_train_B.glob("*.jpg"))

files_B_test = list(path_test_B.glob("**/*.jpg"))
if not files_B_test: files_B_test = list(path_test_B.glob("*.jpg"))

processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

# ==========================================
# 4. PARTIE 1 : CALCUL DU RANDOM ROBUSTE (LIVE)
# ==========================================
print(f"\n>>> Calcul de la Baseline Random ({N_ROUNDS_RANDOM} rounds)...")
# Matrice pour stocker les résultats Random : (Rounds, Steps)
random_results_matrix = np.zeros((N_ROUNDS_RANDOM, len(BUDGET_WITH_ZERO)))

for round_idx in range(N_ROUNDS_RANDOM):
    print(f"   Round {round_idx + 1}/{N_ROUNDS_RANDOM}")
    
    # 1. Reset Seed & Pool
    current_seed = 42 + round_idx 
    random.seed(current_seed)
    pool_copy = files_B_pool.copy()
    random.shuffle(pool_copy)
    labeled_B = []
    
    # 2. Step 0 (Baseline)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    mse0 = evaluate_mse(model, test_loader)
    random_results_matrix[round_idx, 0] = mse0
    
    # 3. Steps suivants
    for step_idx, pct in enumerate(BUDGET_STEPS):
        res_idx = step_idx + 1
        
        # Sélection Random
        target_count = int(len(pool_copy) * (pct / 100.0))
        current_count = len(labeled_B)
        needed = target_count - current_count
        if needed > 0:
            labeled_B.extend(pool_copy[current_count : target_count])
            
        # Entraînement Rapide (On ne sauvegarde pas le modèle sur disque pour aller vite)
        mixed_files = files_A + labeled_B
        train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
        
        model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=AL_LR)
        criterion = nn.MSELoss()
        
        # Boucle d'entraînement silencieuse
        for epoch in range(AL_EPOCHS):
            for pix, tgt in train_loader:
                pix, tgt = pix.to(device), tgt.to(device)
                optimizer.zero_grad()
                loss = criterion(torch.sigmoid(model(pix).logits), tgt)
                loss.backward()
                optimizer.step()
        
        # Evaluation
        mse = evaluate_mse(model, test_loader)
        random_results_matrix[round_idx, res_idx] = mse

# Calcul des stats Random
rand_means = np.mean(random_results_matrix, axis=0)
rand_stds = np.std(random_results_matrix, axis=0)

print(" -> Random calculé.")

# ==========================================
# 5. PARTIE 2 : CHARGEMENT MC DROPOUT
# ==========================================
print("\n>>> Chargement des résultats MC Dropout existants...")
mcdrop_scores = []

# Step 0 (On prend la même baseline que Random moyenne pour être cohérent, ou on recalcule)
# Ici on prend la moyenne Random 0% comme point de départ commun
mcdrop_scores.append(rand_means[0]) 

for pct in BUDGET_STEPS:
    folder_name = f"model_mcdropout_{pct}percent"
    ckpt_path = path_mcdrop_checkpoints / folder_name
    
    if ckpt_path.exists():
        try:
            model = ResNetForImageClassification.from_pretrained(ckpt_path).to(device)
            mse = evaluate_mse(model, test_loader)
            mcdrop_scores.append(mse)
            print(f"   {pct}% : {mse:.6f}")
        except:
            mcdrop_scores.append(None)
            print(f"   {pct}% : Erreur chargement")
    else:
        mcdrop_scores.append(None)
        print(f"   {pct}% : Introuvable")

# ==========================================
# 6. GRAPHIQUE COMPARATIF
# ==========================================
plt.figure(figsize=(12, 7))

# A. Courbe Random avec Zone d'Ombre (Variance) 
plt.plot(BUDGET_WITH_ZERO, rand_means, color='gray', linestyle='--', label='Random Strategy (Moyenne)')
# Remplissage de l'écart-type (Zone de confiance)
plt.fill_between(BUDGET_WITH_ZERO, 
                 rand_means - rand_stds, 
                 rand_means + rand_stds, 
                 color='gray', alpha=0.2, label='Random (Variance)')

# B. Courbe MC Dropout
# On filtre les None au cas où
valid_steps = [BUDGET_WITH_ZERO[i] for i in range(len(mcdrop_scores)) if mcdrop_scores[i] is not None]
valid_scores = [s for s in mcdrop_scores if s is not None]

plt.plot(valid_steps, valid_scores, marker='s', color='purple', linewidth=3, label='Uncertainty (MC Dropout)')

# Décoration
plt.title("Comparaison Active Learning : Robustesse vs Stratégie\n(Random calculé sur 3 rounds)")
plt.xlabel("% de données Domain B annotées")
plt.ylabel("MSE Loss (Test Domain B)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(BUDGET_WITH_ZERO)

save_path = output_dir / "compare_robust_vs_mcdropout.png"
plt.savefig(save_path)
print(f"\nGraphique sauvegardé : {save_path}")
plt.show()

# Tableau Console
print("\n--- TABLEAU RÉCAPITULATIF ---")
print(f"{'PCT':<5} | {'RANDOM (Moy)':<12} | {'MC DROP':<12} | {'DIFF':<10}")
print("-" * 50)
for i, pct in enumerate(BUDGET_WITH_ZERO):
    r_mean = rand_means[i]
    mc = mcdrop_scores[i]
    
    if mc is not None:
        # Si MC Drop est en dessous de (Moyenne - EcartType), c'est une victoire significative
        diff = r_mean - mc
        sig = "*" if mc < (r_mean - rand_stds[i]) else "" 
        print(f"{pct:<5} | {r_mean:.6f}       | {mc:.6f}       | {diff:+.6f} {sig}")
    else:
        print(f"{pct:<5} | {r_mean:.6f}       | N/A")

print("\n(Note : Une étoile '*' signifie que MC Dropout est significativement meilleur que le Random)")