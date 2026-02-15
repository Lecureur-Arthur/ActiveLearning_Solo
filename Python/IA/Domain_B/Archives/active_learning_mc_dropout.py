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

# Chemins
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"

# Modèle de base (A)
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Dossier de résultats
results_dir = script_dir / "AL_Results" / "Uncertainty_MC_Dropout"
checkpoints_dir = results_dir / "checkpoints"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

MAX_X = 1920.0
MAX_Y = 1080.0

# Paramètres
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 5      
AL_LR = 1e-5       
MC_DROPOUT_ITERATIONS = 10  # Nombre de prédictions par image pour estimer l'incertitude

print(f"--- ACTIVE LEARNING (UNCERTAINTY - MC DROPOUT) ---")

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
# 3. FONCTION DE SÉLECTION (MC DROPOUT)
# ==========================================
def select_samples_mc_dropout(model, pool_files, n_needed, processor):
    """
    Sélectionne les images où le modèle a la plus grande variance (incertitude)
    en faisant plusieurs passes avec Dropout activé.
    """
    print(f" -> Analyse de l'incertitude sur {len(pool_files)} images (MC Dropout)...")
    
    # On force le mode TRAIN pour garder le Dropout actif, 
    # mais on gèle les Batch Norm pour la stabilité
    model.train() 
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    uncertainties = []
    
    # Création d'un loader temporaire pour le pool (sans shuffle)
    pool_loader = DataLoader(ALDataset(pool_files, processor), batch_size=16, shuffle=False)
    
    with torch.no_grad():
        for pix, _ in tqdm(pool_loader, desc="Calcul Incertitude"):
            pix = pix.to(device)
            
            # Stocker les prédictions multiples pour ce batch
            batch_preds = []
            
            # On fait X passages
            for _ in range(MC_DROPOUT_ITERATIONS):
                outputs = torch.sigmoid(model(pix).logits) # (Batch, 2)
                batch_preds.append(outputs.cpu().numpy())
            
            # batch_preds shape: (Iterations, Batch, 2)
            batch_preds = np.array(batch_preds)
            
            # Calcul de la variance pour chaque image du batch
            # On moyenne la variance sur X et Y pour avoir un seul score
            # Shape finale : (Batch,)
            batch_variance = np.var(batch_preds, axis=0).mean(axis=1)
            
            uncertainties.extend(batch_variance)
            
    # Conversion en numpy
    uncertainties = np.array(uncertainties)
    
    # On veut les indices avec la PLUS GRANDE variance
    # argsort trie du plus petit au plus grand, donc on prend la fin
    sorted_indices = np.argsort(uncertainties)[::-1]
    top_indices = sorted_indices[:n_needed]
    
    selected_files = [pool_files[i] for i in top_indices]
    
    # On retire les fichiers sélectionnés du pool
    remaining_pool = [pool_files[i] for i in range(len(pool_files)) if i not in top_indices]
    
    return selected_files, remaining_pool, uncertainties.mean()


# ==========================================
# 4. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
if not files_B_pool: files_B_pool = list(path_train_B.glob("*.jpg"))

files_B_test = list(path_test_B.glob("**/*.jpg"))
if not files_B_test: files_B_test = list(path_test_B.glob("*.jpg"))

processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

# Initialisation
labeled_B = []
results_mse = []
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS

# Pour MC Dropout, on ne mélange PAS le pool initialement, 
# car on va le trier par incertitude à chaque étape.

# ==========================================
# 5. BASELINE (0%)
# ==========================================
print("\n--- BASELINE (0%) ---")
model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
initial_mse = evaluate_mse(model, test_loader)
results_mse.append(initial_mse)
print(f" -> MSE Initiale : {initial_mse:.6f}")

# ==========================================
# 6. BOUCLE ACTIVE LEARNING
# ==========================================
current_pool = files_B_pool # Copie de travail

for pct in BUDGET_STEPS:
    print(f"\n>>> VISÉE : {pct}% du Domain B (Stratégie: MC Dropout)")
    
    # 1. Calcul du nombre à ajouter
    target_total = int(len(files_B_pool) * (pct / 100.0))
    current_count = len(labeled_B)
    n_needed = target_total - current_count
    
    if n_needed > 0:
        # --- C'EST ICI QUE LA MAGIE OPÈRE ---
        # On utilise le modèle actuel pour choisir les images difficiles
        print(" -> Sélection des images les plus incertaines...")
        new_data, current_pool, avg_uncert = select_samples_mc_dropout(model, current_pool, n_needed, processor)
        
        labeled_B.extend(new_data)
        print(f" -> Ajout de {len(new_data)} images (Incertitude moy: {avg_uncert:.6f})")
    
    # 2. Dataset Mixte
    mixed_files = files_A + labeled_B
    train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
    
    # 3. Rechargement Modèle A (Reset pour Fine-Tuning équitable)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model.train()
    
    # 4. Fine-Tuning
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
            
    # 5. Evaluation
    current_mse = evaluate_mse(model, test_loader)
    results_mse.append(current_mse)
    print(f" -> MSE Résultat : {current_mse:.6f}")
    
    # 6. Sauvegarde
    folder_name = f"model_mcdropout_{pct}percent"
    save_path = checkpoints_dir / folder_name
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

# ==========================================
# 7. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(BUDGET_WITH_ZERO, results_mse, 'o-', linewidth=2, color='purple', label='MC Dropout Strategy')
plt.title("Active Learning Curve (Incertitude)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

out_file = results_dir / "mcdropout_curve.png"
plt.savefig(out_file)
print(f"\nTerminé ! Graphique : {out_file}")
plt.show()

# Tableau final
print("\n--- RÉSUMÉ MC DROPOUT ---")
for p, m in zip(BUDGET_WITH_ZERO, results_mse):
    print(f"{p}% -> MSE: {m:.6f}")