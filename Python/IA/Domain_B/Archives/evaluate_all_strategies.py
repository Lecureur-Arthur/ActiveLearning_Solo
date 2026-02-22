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
# 1. CONFIGURATION GLOBALE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"
results_dir = script_dir / "AL_Results_Automated"

MAX_X, MAX_Y = 1920.0, 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
N_ROUNDS_RANDOM = 3

print("="*60)
print("📊 PHASE 2 : ÉVALUATION ET GÉNÉRATION DES GRAPHIQUES")
print("="*60)

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
    if not os.path.exists(model_path): return None
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
    except: return None

processor = AutoImageProcessor.from_pretrained(model_A_path)
files_B_test = list(path_test_B.glob("**/*.jpg"))
test_loader = DataLoader(TestDataset(files_B_test, processor), batch_size=32, shuffle=False)

# --- 1. EVALUATION BASELINE (0%) ---
print("-> Calcul de la Baseline Initiale (0%)...")
base_mse = evaluate_mse(model_A_path, test_loader)
global_results = {}

# --- 2. EVALUATION RANDOM ROBUST ---
print("-> Évaluation du Random Robust...")
rand_matrix = np.zeros((N_ROUNDS_RANDOM, len(BUDGET_WITH_ZERO)))
for r in range(N_ROUNDS_RANDOM):
    rand_matrix[r, 0] = base_mse
    for step_idx, pct in enumerate(BUDGET_STEPS):
        path = results_dir / "Random" / f"Round_{r+1}" / f"model_{pct}pct"
        rand_matrix[r, step_idx+1] = evaluate_mse(path, test_loader)

rand_mean = np.mean(rand_matrix, axis=0)
rand_std = np.std(rand_matrix, axis=0)
global_results["Random"] = rand_mean

def plot_vs_random(strat_name, strat_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(BUDGET_WITH_ZERO, rand_mean, color='gray', linestyle='--', label='Random (Moyenne)')
    plt.fill_between(BUDGET_WITH_ZERO, rand_mean - rand_std, rand_mean + rand_std, color='gray', alpha=0.2)
    plt.plot(BUDGET_WITH_ZERO, strat_scores, marker='o', linewidth=2.5, label=strat_name)
    plt.title(f"{strat_name} vs Random Baseline\n(Test sur Dossiers 9, 10, 11)")
    plt.xlabel("% Domain B")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xticks(BUDGET_WITH_ZERO)
    plt.savefig(results_dir / f"Compare_{strat_name.replace(' ', '_')}.png")
    plt.close()

# --- 3. EVALUATION DES STRATEGIES INTELLIGENTES ---
strategies = {
    "MC Dropout": "MC_Dropout",
    "Learning Loss": "LearningLoss",
    "K-Means": "KMeans",
    "Outliers": "Outliers"
}

for name, folder in strategies.items():
    print(f"-> Évaluation de {name}...")
    scores = [base_mse]
    for pct in tqdm(BUDGET_STEPS, desc=name, leave=False):
        path = results_dir / folder / f"model_{pct}pct"
        mse = evaluate_mse(path, test_loader)
        scores.append(mse if mse is not None else np.nan)
    
    global_results[name] = scores
    plot_vs_random(name, scores)

# --- 4. GRAPHIQUE FINAL ---
print("-> Génération du Graphique Global...")
plt.figure(figsize=(12, 8))
plt.plot(BUDGET_WITH_ZERO, rand_mean, color='gray', linestyle='--', linewidth=2, label='Random (Moy)')
plt.fill_between(BUDGET_WITH_ZERO, rand_mean - rand_std, rand_mean + rand_std, color='gray', alpha=0.15)

colors = {"MC Dropout": "purple", "Learning Loss": "orange", "K-Means": "green", "Outliers": "red"}
markers = {"MC Dropout": "s", "Learning Loss": "^", "K-Means": "d", "Outliers": "x"}

for name, scores in global_results.items():
    if name == "Random": continue
    plt.plot(BUDGET_WITH_ZERO, scores, marker=markers[name], color=colors[name], linewidth=2.5, label=name)

plt.title("Comparaison Finale des Stratégies d'Active Learning\n(Test sur Séquences Temporelles Inconnues)")
plt.xlabel("% Domain B Annoté")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.xticks(BUDGET_WITH_ZERO)
plt.savefig(results_dir / "GRAND_FINAL_BENCHMARK.png")

print(f"\n✅ TOUT EST TERMINE ! Les graphiques sont dans : {results_dir}")