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

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

base_results_dir = script_dir / "AL_Results"
output_graphs_dir = base_results_dir / "Graphiques_Finaux"
os.makedirs(output_graphs_dir, exist_ok=True)

MAX_X, MAX_Y = 1920.0, 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
N_ROUNDS_RANDOM = 3

print("="*60)
print("📊 ÉVALUATION : STRATÉGIES MIXTES VS RANDOM")
print("="*60)

# ==========================================
# 2. DATASET ET FONCTION D'ÉVALUATION
# ==========================================
class TestDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        try:
            x = float(path.stem.split('_')[-2]) / MAX_X
            y = float(path.stem.split('_')[-1]) / MAX_Y
        except: x, y = 0.0, 0.0
        return self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0), torch.tensor([x, y], dtype=torch.float32)

def evaluate_mse(model_path, test_loader):
    if not os.path.exists(model_path): 
        return None
    try:
        model = ResNetForImageClassification.from_pretrained(model_path).to(device)
        model.eval()
        total_loss = 0.0
        criterion = nn.MSELoss()
        with torch.no_grad():
            for pix, tgt in test_loader:
                out = torch.sigmoid(model(pix.to(device)).logits)
                total_loss += criterion(out, tgt.to(device)).item()
        return total_loss / len(test_loader)
    except Exception as e:
        print(f"Erreur évaluation {model_path}: {e}")
        return None

processor = AutoImageProcessor.from_pretrained(model_A_path)
files_B_test = list(path_test_B.glob("**/*.jpg"))
test_loader = DataLoader(TestDataset(files_B_test, processor), batch_size=32, shuffle=False)

all_scores = {}

# ==========================================
# 3. CALCUL DE LA BASELINE (0%)
# ==========================================
print("\n-> Calcul de la Baseline (0%)...")
base_mse = evaluate_mse(model_A_path, test_loader)
print(f"   MSE Baseline : {base_mse:.6f}")

# ==========================================
# 4. RÉCUPÉRATION DU RANDOM ROBUST
# ==========================================
print("\n-> Évaluation du Random Robust...")
rand_matrix = np.zeros((N_ROUNDS_RANDOM, len(BUDGET_WITH_ZERO)))

for r in range(N_ROUNDS_RANDOM):
    rand_matrix[r, 0] = base_mse
    for step_idx, pct in enumerate(BUDGET_STEPS):
        path = base_results_dir / "Random_Robust" / f"Round_{r+1}" / f"model_{pct}pct"
        mse = evaluate_mse(path, test_loader)
        rand_matrix[r, step_idx+1] = mse if mse is not None else np.nan

rand_mean = np.nanmean(rand_matrix, axis=0)
rand_std = np.nanstd(rand_matrix, axis=0)

# ==========================================
# 5. RÉCUPÉRATION DES STRATÉGIES MIXTES
# ==========================================
mixed_strategies = {
    "Sequential (Dropout -> KMeans)": ("Mixed_Sequential", "model_sequential"),
    "Integrated (50% Uncert / 50% Div)": ("Mixed_Integrated", "model_integrated")
}

for name, (folder, prefix) in mixed_strategies.items():
    print(f"-> Évaluation de {name}...")
    scores = [base_mse]
    for pct in tqdm(BUDGET_STEPS, desc=name, leave=False):
        path = base_results_dir / folder / "checkpoints" / f"{prefix}_{pct}percent"
        mse = evaluate_mse(path, test_loader)
        scores.append(mse if mse is not None else np.nan)
    all_scores[name] = scores

# ==========================================
# 6. GÉNÉRATION DU GRAPHIQUE
# ==========================================
print("\n-> Génération du graphique final...")
plt.figure(figsize=(12, 8))

# Courbe Random
plt.plot(BUDGET_WITH_ZERO, rand_mean, color='gray', linestyle='--', linewidth=2, label='Random (Moyenne)')
plt.fill_between(BUDGET_WITH_ZERO, rand_mean - rand_std, rand_mean + rand_std, color='gray', alpha=0.15)

# Courbes Mixtes
colors = {"Sequential (Dropout -> KMeans)": "teal", "Integrated (50% Uncert / 50% Div)": "magenta"}
markers = {"Sequential (Dropout -> KMeans)": "s", "Integrated (50% Uncert / 50% Div)": "*"}

for strat_name, scores in all_scores.items():
    if not np.isnan(scores).all():
        plt.plot(BUDGET_WITH_ZERO, scores, 
                 marker=markers[strat_name], color=colors[strat_name], 
                 linewidth=2.5, markersize=10 if strat_name.startswith("Integrated") else 8, 
                 label=strat_name)

plt.title("Performances des Stratégies Mixtes (Hybrides) vs Random", fontsize=14)
plt.xlabel("% de données Domain B annotées", fontsize=12)
plt.ylabel("MSE Loss (Test sur Séquences inédites)", fontsize=12)
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend(fontsize=11)
plt.xticks(BUDGET_WITH_ZERO)

save_path = output_graphs_dir / "4_Graph_Mixed_vs_Random.png"
plt.savefig(save_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\n✅ TERMINÉ ! Graphique généré ici : {save_path}")

# Affichage des scores dans la console pour votre rapport
print("\n--- RÉSUMÉ DES RÉSULTATS MIXTES ---")
for strat_name, scores in all_scores.items():
    print(f"\n{strat_name} :")
    for p, m in zip(BUDGET_WITH_ZERO, scores):
        print(f"  {p}% -> MSE: {m:.6f}")