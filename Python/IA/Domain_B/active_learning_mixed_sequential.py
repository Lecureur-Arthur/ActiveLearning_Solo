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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

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

results_dir = script_dir / "AL_Results" / "Mixed_Sequential"
checkpoints_dir = results_dir / "checkpoints"
os.makedirs(checkpoints_dir, exist_ok=True)

MAX_X, MAX_Y = 1920.0, 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
AL_EPOCHS = 5      
AL_LR = 1e-5
MC_ITERATIONS = 10

print("="*60)
print("🧬 ACTIVE LEARNING : MIXED SEQUENTIAL (Dropout -> KMeans)")
print("="*60)

# ==========================================
# 2. DATASET ET UTILITAIRES
# ==========================================
class ALDataset(Dataset):
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

def evaluate_mse(model, loader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for pix, tgt in loader:
            out = torch.sigmoid(model(pix.to(device)).logits)
            total_loss += criterion(out, tgt.to(device)).item()
    return total_loss / len(loader)

def get_features(model, file_list, processor):
    if not file_list: return np.array([])
    model.eval()
    loader = DataLoader(ALDataset(file_list, processor), batch_size=64, shuffle=False)
    feats = []
    with torch.no_grad():
        for pix, _ in loader:
            outs = model.resnet(pix.to(device))
            feats.append(outs.pooler_output.squeeze().cpu().numpy())
    return np.concatenate(feats, axis=0)

def get_mc_dropout_uncertainty(model, file_list, processor):
    model.train() 
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d): module.eval()
    loader = DataLoader(ALDataset(file_list, processor), batch_size=32, shuffle=False)
    uncertainties = []
    with torch.no_grad():
        for pix, _ in tqdm(loader, desc="   ↳ Calcul Incertitude", leave=False):
            preds = np.array([torch.sigmoid(model(pix.to(device)).logits).cpu().numpy() for _ in range(MC_ITERATIONS)])
            uncertainties.extend(np.var(preds, axis=0).mean(axis=1))
    return np.array(uncertainties)

# ==========================================
# 3. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
files_B_test = list(path_test_B.glob("**/*.jpg"))
processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

print("\n-> Calcul de la Baseline (0%)...")
model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
base_mse = evaluate_mse(model, test_loader)
results_mse = [base_mse]
labeled_seq = []
pool_seq = files_B_pool.copy()

# ==========================================
# 4. BOUCLE ACTIVE LEARNING
# ==========================================
for pct in BUDGET_STEPS:
    print(f"\n>>> VISÉE : {pct}% (Sequential)")
    target_total = int(len(files_B_pool) * (pct / 100.0))
    n_needed = target_total - len(labeled_seq)
    
    if n_needed > 0:
        model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
        
        # Étape 1 : MC Dropout
        n_candidates = min(n_needed * 3, len(pool_seq))
        uncerts = get_mc_dropout_uncertainty(model, pool_seq, processor)
        candidate_idx = np.argsort(uncerts)[::-1][:n_candidates]
        candidates = [pool_seq[i] for i in candidate_idx]
        
        # Étape 2 : K-Means
        print("   ↳ K-Means sur le sous-groupe incertain...")
        feats = get_features(model, candidates, processor)
        kmeans = KMeans(n_clusters=n_needed, random_state=42, n_init=10).fit(feats)
        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats)
        
        top_idx = list(np.unique(closest))
        while len(top_idx) < n_needed:
            remain = [i for i in range(len(candidates)) if i not in top_idx]
            top_idx.append(remain[0])
            
        selected_files = [candidates[i] for i in top_idx]
        labeled_seq.extend(selected_files)
        pool_seq = [f for f in pool_seq if f not in selected_files]

    # Entraînement
    train_loader = DataLoader(ALDataset(files_A + labeled_seq, processor), batch_size=32, shuffle=True)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=AL_LR)
    criterion = nn.MSELoss()
    
    for epoch in range(AL_EPOCHS):
        for pix, tgt in tqdm(train_loader, desc=f"   ↳ Train {pct}%", leave=False):
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(model(pix.to(device)).logits), tgt.to(device))
            loss.backward()
            optimizer.step()
            
    mse = evaluate_mse(model, test_loader)
    results_mse.append(mse)
    print(f" -> MSE Résultat : {mse:.6f}")
    
    save_path = checkpoints_dir / f"model_sequential_{pct}percent"
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)

# ==========================================
# 5. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(BUDGET_WITH_ZERO, results_mse, 's-', linewidth=2, color='teal', label='Sequential (MC Dropout -> K-Means)')
plt.title("Active Learning Curve (Mixed Sequential)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

out_file = results_dir / "mixed_sequential_curve.png"
plt.savefig(out_file)
print(f"\nTerminé ! Graphique : {out_file}")