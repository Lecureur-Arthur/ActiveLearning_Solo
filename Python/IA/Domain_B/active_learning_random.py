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

# Chemins relatifs
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

# Sources
path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"

# Modèle de base (A)
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Dossier de résultats
results_dir = script_dir / "AL_Results" / "Random_Strategy"
os.makedirs(results_dir, exist_ok=True)

# Dossier pour les checkpoints (sauvegardes intermédiaires)
checkpoints_dir = results_dir / "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

MAX_X = 1920.0
MAX_Y = 1080.0

# Paramètres Active Learning
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 5      
AL_LR = 1e-5       

print(f"--- ACTIVE LEARNING (RANDOM + SAVE ALL) ---")
print(f"Modèle Base : {model_A_path}")

if not model_A_path.exists():
    raise FileNotFoundError(f"Le modèle A n'existe pas : {model_A_path}")

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

print(f"Data A : {len(files_A)}")
print(f"Pool B : {len(files_B_pool)}")

processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

random.seed(42)
random.shuffle(files_B_pool)

labeled_B = []
results_mse = []
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS

# ==========================================
# 4. STEP 0 (BASELINE)
# ==========================================
print("\n--- BASELINE (0%) ---")
model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
initial_mse = evaluate_mse(model, test_loader)
results_mse.append(initial_mse)
print(f" -> MSE Initiale : {initial_mse:.6f}")

# ==========================================
# 5. BOUCLE ACTIVE LEARNING
# ==========================================
for pct in BUDGET_STEPS:
    print(f"\n>>> VISÉE : {pct}% du Domain B")
    
    # Sélection Random
    target_count = int(len(files_B_pool) * (pct / 100.0))
    current_count = len(labeled_B)
    needed = target_count - current_count
    
    if needed > 0:
        new_data = files_B_pool[current_count : target_count]
        labeled_B.extend(new_data)
        print(f" -> Ajout de {len(new_data)} images.")
    
    # Dataset Mixte
    mixed_files = files_A + labeled_B
    train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
    
    # Rechargement Modèle A
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model.train()
    
    # Fine-Tuning
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
    
    # --- SAUVEGARDE SYSTÉMATIQUE À CHAQUE PALIER ---
    folder_name = f"model_random_{pct}percent"
    save_path = checkpoints_dir / folder_name
    
    # Création du dossier s'il n'existe pas
    os.makedirs(save_path, exist_ok=True)
    
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    print(f" -> Modèle sauvegardé dans : {save_path}")
    # -----------------------------------------------

# ==========================================
# 6. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(BUDGET_WITH_ZERO, results_mse, 'o-', linewidth=2, label='Random Strategy')
plt.title("Active Learning Curve (Avec Sauvegarde des modèles)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

out_file = results_dir / "random_strategy_curve.png"
plt.savefig(out_file)
print(f"\nTerminé ! Graphique : {out_file}")
plt.show()

# Tableau final
print("\n--- RÉSUMÉ ---")
for p, m in zip(BUDGET_WITH_ZERO, results_mse):
    print(f"{p}% -> MSE: {m:.6f}")