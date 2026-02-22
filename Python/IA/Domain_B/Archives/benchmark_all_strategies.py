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

# Chemins Données
path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Chemins des Checkpoints (Vos résultats sauvegardés)
results_root = script_dir / "AL_Results"
paths_strategies = {
    "MC Dropout":      results_root / "Uncertainty_MC_Dropout" / "checkpoints",
    "Learning Loss":   results_root / "Uncertainty_LearningLoss" / "checkpoints",
    "K-Means":         results_root / "Diversity_KMeans" / "checkpoints",
    "Outliers":        results_root / "Diversity_Outliers" / "checkpoints"
}

# Dossier de sortie
output_dir = script_dir / "AL_Results" / "FINAL_BENCHMARK"
os.makedirs(output_dir, exist_ok=True)

# Paramètres
MAX_X = 1920.0
MAX_Y = 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
AL_EPOCHS = 5      
AL_LR = 1e-5 
N_ROUNDS_RANDOM = 3  # Nombre de répétitions pour la courbe Random

print("--- GRAND COMPARATIF ACTIVE LEARNING ---")

# ==========================================
# 2. DATASET
# ==========================================
class ALDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        name = os.path.basename(path).rsplit('.', 1)[0]
        parts = name.split('_')
        try:
            x = float(parts[-2]) / MAX_X
            y = float(parts[-1]) / MAX_Y
        except: x, y = 0.0, 0.0
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
            total_loss += criterion(out, tgt).item()
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
# 4. REFERENCE : ROBUST RANDOM (CALCUL LIVE)
# ==========================================
print(f"\n1. Calcul de la Baseline RANDOM ({N_ROUNDS_RANDOM} rounds)...")
rand_matrix = np.zeros((N_ROUNDS_RANDOM, len(BUDGET_WITH_ZERO)))

for r in range(N_ROUNDS_RANDOM):
    print(f"   Round {r+1}...")
    random.seed(42 + r)
    pool_copy = files_B_pool.copy()
    random.shuffle(pool_copy)
    labeled = []
    
    # Step 0
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    rand_matrix[r, 0] = evaluate_mse(model, test_loader)
    
    for step_idx, pct in enumerate(BUDGET_STEPS):
        target = int(len(pool_copy) * (pct / 100.0))
        needed = target - len(labeled)
        if needed > 0: labeled.extend(pool_copy[len(labeled):target])
        
        # Train
        train_ds = ALDataset(files_A + labeled, processor)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
        model.train()
        optim = torch.optim.AdamW(model.parameters(), lr=AL_LR)
        crit = nn.MSELoss()
        
        for _ in range(AL_EPOCHS):
            for p, t in train_loader:
                p, t = p.to(device), t.to(device)
                optim.zero_grad()
                loss = crit(torch.sigmoid(model(p).logits), t)
                loss.backward()
                optim.step()
        
        rand_matrix[r, step_idx+1] = evaluate_mse(model, test_loader)

rand_mean = np.mean(rand_matrix, axis=0)
rand_std = np.std(rand_matrix, axis=0)
print(" -> Random terminé.")

# ==========================================
# 5. CHARGEMENT DES AUTRES STRATÉGIES
# ==========================================
print("\n2. Chargement des stratégies avancées...")

strategies_results = {} # Dictionnaire pour stocker les résultats

# Préfixes des dossiers selon la stratégie (pour retrouver les fichiers)
prefixes = {
    "MC Dropout": "model_mcdropout_",
    "Learning Loss": "model_learningloss_",
    "K-Means": "model_kmeans_",
    "Outliers": "model_outliers_"
}

for strat_name, strat_path in paths_strategies.items():
    print(f"   -> {strat_name}")
    scores = []
    # Step 0 : On utilise la moyenne Random 0% comme point de départ commun
    scores.append(rand_mean[0])
    
    folder_prefix = prefixes[strat_name]
    
    for pct in BUDGET_STEPS:
        folder_name = f"{folder_prefix}{pct}percent"
        full_path = strat_path / folder_name
        
        if full_path.exists():
            try:
                model = ResNetForImageClassification.from_pretrained(full_path).to(device)
                mse = evaluate_mse(model, test_loader)
                scores.append(mse)
            except:
                scores.append(None)
                print(f"      ⚠️ Erreur chargement {pct}%")
        else:
            scores.append(None)
            print(f"      ⚠️ Manquant : {folder_name}")
            
    strategies_results[strat_name] = scores

# ==========================================
# 6. GRAPHIQUE FINAL
# ==========================================
plt.figure(figsize=(12, 8))

# A. Zone Random
plt.plot(BUDGET_WITH_ZERO, rand_mean, color='gray', linestyle='--', linewidth=2, label='Random (Baseline)')
plt.fill_between(BUDGET_WITH_ZERO, rand_mean - rand_std, rand_mean + rand_std, color='gray', alpha=0.15)

# B. Stratégies
colors = {
    "MC Dropout": "purple", 
    "Learning Loss": "orange", 
    "K-Means": "green", 
    "Outliers": "red"
}
markers = {
    "MC Dropout": "s", 
    "Learning Loss": "^", 
    "K-Means": "d", 
    "Outliers": "x"
}

for name, scores in strategies_results.items():
    # Filtrer les None
    valid_x = [BUDGET_WITH_ZERO[i] for i in range(len(scores)) if scores[i] is not None]
    valid_y = [s for s in scores if s is not None]
    
    if len(valid_y) > 1:
        plt.plot(valid_x, valid_y, 
                 marker=markers[name], 
                 color=colors[name], 
                 linewidth=2.5, 
                 label=name)

plt.title("Comparaison Globale des Stratégies d'Active Learning\n(MSE Loss sur Domain B)")
plt.xlabel("% de données Domain B annotées")
plt.ylabel("MSE Loss (Plus bas = Mieux)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.xticks(BUDGET_WITH_ZERO)

save_path = output_dir / "FINAL_BENCHMARK_GRAPH.png"
plt.savefig(save_path)
print(f"\n✅ Graphique sauvegardé : {save_path}")
plt.show()

# ==========================================
# 7. TABLEAU RECAPITULATIF
# ==========================================
print("\n--- TABLEAU FINAL (MSE LOSS) ---")
header = f"{'PCT':<5} | {'Random (Ref)':<12}"
for name in strategies_results.keys():
    header += f" | {name[:10]:<10}"
print(header)
print("-" * len(header))

for i, pct in enumerate(BUDGET_WITH_ZERO):
    row = f"{pct:<5} | {rand_mean[i]:.6f}      "
    for name in strategies_results.keys():
        val = strategies_results[name][i]
        str_val = f"{val:.6f}" if val is not None else "N/A"
        
        # Petit indicateur si meilleur que random
        if val is not None and val < (rand_mean[i] - rand_std[i]):
            str_val += "*"
        else:
            str_val += " "
            
        row += f" | {str_val:<10}"
    print(row)
    
print("\n(* : Performance significativement meilleure que le hasard)")