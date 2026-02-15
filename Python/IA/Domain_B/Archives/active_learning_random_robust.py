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

path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

results_dir = script_dir / "AL_Results" / "Random_Robust"
os.makedirs(results_dir, exist_ok=True)

MAX_X = 1920.0
MAX_Y = 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 5      
AL_LR = 1e-5 
N_ROUNDS = 3  # Nombre de fois qu'on répète l'expérience pour avoir les barres d'erreur

# ... (Classe Dataset et Fonctions utilitaires identiques aux précédents scripts) ...
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
# 2. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
if not files_B_pool: files_B_pool = list(path_train_B.glob("*.jpg"))
files_B_test = list(path_test_B.glob("**/*.jpg"))
if not files_B_test: files_B_test = list(path_test_B.glob("*.jpg"))

processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS

# Stockage global : Matrice de taille (N_ROUNDS, Nombre d'étapes)
all_rounds_results = np.zeros((N_ROUNDS, len(BUDGET_WITH_ZERO)))

print(f"--- ROBUST RANDOM ({N_ROUNDS} Rounds) ---")

# ==========================================
# 3. BOUCLE DES ROUNDS (Répétition)
# ==========================================
for round_idx in range(N_ROUNDS):
    print(f"\n=== ROUND {round_idx + 1}/{N_ROUNDS} ===")
    
    # IMPORTANT : Changer la seed à chaque round pour avoir un "Hasard" différent
    current_seed = 42 + round_idx 
    random.seed(current_seed)
    
    # On remélange le pool différemment à chaque round
    pool_copy = files_B_pool.copy()
    random.shuffle(pool_copy)
    
    labeled_B = []
    
    # --- Etape 0 ---
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    mse0 = evaluate_mse(model, test_loader)
    all_rounds_results[round_idx, 0] = mse0
    print(f"  Step 0% : {mse0:.5f}")
    
    # --- Etapes suivantes ---
    for step_idx, pct in enumerate(BUDGET_STEPS):
        # Index dans le tableau de résultats (+1 car on a mis le 0% au début)
        res_idx = step_idx + 1 
        
        target_count = int(len(pool_copy) * (pct / 100.0))
        current_count = len(labeled_B)
        needed = target_count - current_count
        
        if needed > 0:
            new_data = pool_copy[current_count : target_count] # Prend les suivants dans le désordre
            labeled_B.extend(new_data)
        
        # Train
        mixed_files = files_A + labeled_B
        train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
        
        model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=AL_LR)
        criterion = nn.MSELoss()
        
        # On réduit un peu l'affichage tqdm pour ne pas spammer
        for epoch in range(AL_EPOCHS):
            for pix, tgt in train_loader:
                pix, tgt = pix.to(device), tgt.to(device)
                optimizer.zero_grad()
                loss = criterion(torch.sigmoid(model(pix).logits), tgt)
                loss.backward()
                optimizer.step()
        
        mse = evaluate_mse(model, test_loader)
        all_rounds_results[round_idx, res_idx] = mse
        print(f"  Step {pct}% : {mse:.5f}")

# ==========================================
# 4. CALCUL STATISTIQUE & GRAPHIQUE
# ==========================================
# Moyenne par colonne (pour chaque étape de budget)
means = np.mean(all_rounds_results, axis=0)
# Ecart-type par colonne
stds = np.std(all_rounds_results, axis=0)

print("\n--- RÉSULTATS MOYENNÉS ---")
for i, pct in enumerate(BUDGET_WITH_ZERO):
    print(f"{pct}% -> Moy: {means[i]:.5f} (+/- {stds[i]:.5f})")

plt.figure(figsize=(10, 6))

# C'est ici qu'on ajoute les barres d'erreur
plt.errorbar(BUDGET_WITH_ZERO, means, yerr=stds, fmt='-o', 
             color='gray', ecolor='lightgray', elinewidth=3, capsize=0, 
             label=f'Random (Moyenne sur {N_ROUNDS} runs)')

plt.title(f"Performance Random Robuste\n(Zone grise = Variabilité du hasard)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.5)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

save_path = results_dir / "robust_random_curve.png"
plt.savefig(save_path)
print(f"Graphique sauvegardé : {save_path}")
plt.show()