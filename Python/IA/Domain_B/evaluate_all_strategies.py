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

# Dossier principal où les autres scripts ont sauvegardé leurs résultats
base_results_dir = script_dir / "AL_Results"
output_graphs_dir = base_results_dir / "Graphiques_Finaux"
os.makedirs(output_graphs_dir, exist_ok=True)

MAX_X, MAX_Y = 1920.0, 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
N_ROUNDS_RANDOM = 3

print("="*60)
print("📊 ÉVALUATION FINALE ET GÉNÉRATION DES GRAPHIQUES")
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

# Préparation du Test Loader
processor = AutoImageProcessor.from_pretrained(model_A_path)
files_B_test = list(path_test_B.glob("**/*.jpg"))
test_loader = DataLoader(TestDataset(files_B_test, processor), batch_size=32, shuffle=False)

# Dictionnaire pour stocker toutes les courbes
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
all_scores["Random"] = rand_mean

# ==========================================
# 5. RÉCUPÉRATION DES AUTRES STRATÉGIES
# ==========================================
# Format : "Nom pour le graph": ("Dossier", "Préfixe du modèle")
strategies_info = {
    "MC Dropout": ("Uncertainty_MC_Dropout", "model_mcdropout"),
    "Learning Loss": ("Uncertainty_LearningLoss", "model_learningloss"),
    "K-Means": ("Diversity_KMeans", "model_kmeans"),
    "Outliers": ("Diversity_Outliers", "model_outliers")
}

for name, (folder, prefix) in strategies_info.items():
    print(f"-> Évaluation de {name}...")
    scores = [base_mse]
    for pct in tqdm(BUDGET_STEPS, desc=name, leave=False):
        path = base_results_dir / folder / "checkpoints" / f"{prefix}_{pct}percent"
        mse = evaluate_mse(path, test_loader)
        scores.append(mse if mse is not None else np.nan)
    all_scores[name] = scores

# ==========================================
# 6. FONCTION DE DESSIN DES GRAPHIQUES
# ==========================================
colors = {"MC Dropout": "purple", "Learning Loss": "darkorange", "K-Means": "green", "Outliers": "darkred"}
markers = {"MC Dropout": "o", "Learning Loss": "^", "K-Means": "d", "Outliers": "X"}

def draw_graph(strategies_to_plot, title, filename):
    plt.figure(figsize=(12, 8))
    
    # Dessiner la courbe Random en fond avec sa zone d'incertitude
    plt.plot(BUDGET_WITH_ZERO, rand_mean, color='gray', linestyle='--', linewidth=2, label='Random (Moyenne)')
    plt.fill_between(BUDGET_WITH_ZERO, rand_mean - rand_std, rand_mean + rand_std, color='gray', alpha=0.15)
    
    # Dessiner les stratégies demandées
    for strat in strategies_to_plot:
        if strat in all_scores and not np.isnan(all_scores[strat]).all():
            plt.plot(BUDGET_WITH_ZERO, all_scores[strat], 
                     marker=markers[strat], color=colors[strat], 
                     linewidth=2.5, markersize=8, label=strat)
            
    plt.title(title, fontsize=14)
    plt.xlabel("% de données Domain B annotées", fontsize=12)
    plt.ylabel("MSE Loss (Test sur Séquences inédites)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend(fontsize=11)
    plt.xticks(BUDGET_WITH_ZERO)
    
    save_path = output_graphs_dir / filename
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"   Graphique généré : {filename}")

# ==========================================
# 7. GÉNÉRATION DES 3 GRAPHIQUES
# ==========================================
print("\n-> Génération des graphiques...")

# 1. GRAPH GLOBAL (Toutes les stratégies)
draw_graph(["MC Dropout", "Learning Loss", "K-Means", "Outliers"], 
           "Comparaison Finale : Toutes les Stratégies d'Active Learning", 
           "1_Graph_Global_Complet.png")

# 2. GRAPH DIVERSITÉ (K-Means & Outliers vs Random)
draw_graph(["K-Means", "Outliers"], 
           "Active Learning par Diversité (K-Means & Outliers) vs Random", 
           "2_Graph_Famille_Diversite.png")

# 3. GRAPH INCERTITUDE (MC Dropout & Learning Loss vs Random)
draw_graph(["MC Dropout", "Learning Loss"], 
           "Active Learning par Incertitude (Dropout & Learning Loss) vs Random", 
           "3_Graph_Famille_Incertitude.png")

print(f"\n✅ TOUT EST TERMINÉ ! Retrouvez vos graphiques dans :\n{output_graphs_dir}")