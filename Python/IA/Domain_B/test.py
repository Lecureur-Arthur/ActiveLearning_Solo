import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

# Dossiers de Test
path_test_A = project_root / "data" / "Split" / "Domain_A" / "test"
path_test_B = project_root / "data" / "Split" / "Domain_B" / "test"

# Modèle de base (0%)
base_model_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Dossier où sont les checkpoints (1%, 2%, etc.)
checkpoints_dir = script_dir / "AL_Results" / "Random_Strategy" / "checkpoints"

# Dossier de sortie pour le graphe comparatif
output_dir = script_dir / "AL_Results" / "Random_Strategy"
os.makedirs(output_dir, exist_ok=True)

# Pourcentages à tester
STEPS = [0, 1, 2, 5, 10, 20, 50]
MAX_X = 1920.0
MAX_Y = 1080.0

print("--- BENCHMARK GLOBAL (A vs B) ---")
print(f"Test A : {path_test_A}")
print(f"Test B : {path_test_B}")

# ==========================================
# 2. DATASET UTILITAIRE
# ==========================================
class BenchmarkDataset(Dataset):
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

def get_mse(model, loader):
    """Calcule la MSE moyenne sur un DataLoader"""
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
# 3. CHARGEMENT DES FICHIERS DE TEST
# ==========================================
files_A = list(path_test_A.glob("**/*.jpg"))
files_B = list(path_test_B.glob("**/*.jpg"))

if not files_A: print("⚠️ Attention: Pas d'images trouvées dans Test A")
if not files_B: print("⚠️ Attention: Pas d'images trouvées dans Test B")

print(f"Images Test A : {len(files_A)}")
print(f"Images Test B : {len(files_B)}")

# ==========================================
# 4. BOUCLE DE TEST SUR CHAQUE MODÈLE
# ==========================================
results_A = []
results_B = []

# On a besoin d'un processeur par défaut pour créer les loaders au début
# On prend celui du modèle de base
default_processor = AutoImageProcessor.from_pretrained(base_model_path)
loader_A = DataLoader(BenchmarkDataset(files_A, default_processor), batch_size=32, shuffle=False)
loader_B = DataLoader(BenchmarkDataset(files_B, default_processor), batch_size=32, shuffle=False)

print("\nLancement des évaluations...")

for pct in STEPS:
    print(f"\n>>> Test du Modèle {pct}%")
    
    # 1. Déterminer quel dossier charger
    if pct == 0:
        current_path = base_model_path
        print(" -> Chargement Modèle Base (Domain A pure)")
    else:
        folder_name = f"model_random_{pct}percent"
        current_path = checkpoints_dir / folder_name
        print(f" -> Chargement Checkpoint : {folder_name}")
    
    if not current_path.exists():
        print(f"❌ ERREUR : Modèle introuvable {current_path}")
        results_A.append(None)
        results_B.append(None)
        continue

    # 2. Charger le modèle
    try:
        model = ResNetForImageClassification.from_pretrained(current_path).to(device)
    except Exception as e:
        print(f"❌ Erreur chargement : {e}")
        continue

    # 3. Évaluer sur A
    mse_a = get_mse(model, loader_A)
    results_A.append(mse_a)
    print(f"    MSE sur Test A : {mse_a:.6f}")

    # 4. Évaluer sur B
    mse_b = get_mse(model, loader_B)
    results_B.append(mse_b)
    print(f"    MSE sur Test B : {mse_b:.6f}")

# ==========================================
# 5. GRAPHIQUE COMPARATIF
# ==========================================
plt.figure(figsize=(10, 6))

# Courbe A (Performance sur l'ancien domaine)
plt.plot(STEPS, results_A, marker='o', label='Test sur Domain A (Ancien)', color='blue', linewidth=2)

# Courbe B (Performance sur le nouveau domaine)
plt.plot(STEPS, results_B, marker='s', label='Test sur Domain B (Nouveau)', color='orange', linewidth=2)

plt.title("Impact du Fine-Tuning : Performance sur A vs B")
plt.xlabel("% de données Domain B ajoutées au train")
plt.ylabel("MSE Loss (Plus bas = Mieux)")
plt.xticks(STEPS)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

# Sauvegarde
out_graph = output_dir / "benchmark_A_vs_B.png"
plt.savefig(out_graph)
print(f"\nGraphique sauvegardé : {out_graph}")

# Tableau Console
print("\n--- TABLEAU RÉCAPITULATIF ---")
print(f"{'Modèle (%)':<12} | {'MSE Test A':<12} | {'MSE Test B':<12}")
print("-" * 42)
for i, pct in enumerate(STEPS):
    val_a = f"{results_A[i]:.5f}" if results_A[i] is not None else "N/A"
    val_b = f"{results_B[i]:.5f}" if results_B[i] is not None else "N/A"
    print(f"{pct:<12} | {val_a:<12} | {val_b:<12}")

plt.show()