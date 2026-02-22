import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import random
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

# ==========================================
# 1. CONFIGURATION GLOBALE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

output_dir = script_dir / "AL_Results_Continuous"
os.makedirs(output_dir, exist_ok=True)

MAX_X, MAX_Y = 1920.0, 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 3 # Moins d'époques car l'apprentissage est continu      
AL_LR = 1e-5 
N_ROUNDS_RANDOM = 3

print("="*60)
print("⚙️ ENTRAÎNEMENT CONTINU (TOUTES LES STRATÉGIES)")
print("="*60)

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

class LossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

def train_standard(model, train_loader, desc_prefix="Train"):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=AL_LR)
    criterion = nn.MSELoss()
    for epoch in range(AL_EPOCHS):
        loop = tqdm(train_loader, desc=f"   ↳ {desc_prefix} [Ep {epoch+1}/{AL_EPOCHS}]", leave=False)
        for pix, tgt in loop:
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(model(pix.to(device)).logits), tgt.to(device))
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f"{loss.item():.4f}")
    return model

files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
processor = AutoImageProcessor.from_pretrained(model_A_path)

# --- 1. RANDOM ROBUST (CONTINU) ---
print("\n--- [1/5] RANDOM ROBUST (CONTINU) ---")
strat_dir_random = output_dir / "Random"
os.makedirs(strat_dir_random, exist_ok=True)

for r in range(N_ROUNDS_RANDOM):
    random.seed(42 + r)
    pool_copy = files_B_pool.copy()
    random.shuffle(pool_copy)
    labeled = []
    round_dir = strat_dir_random / f"Round_{r+1}"
    os.makedirs(round_dir, exist_ok=True)
    
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    pbar = tqdm(BUDGET_STEPS, desc=f"Round Random {r+1}/{N_ROUNDS_RANDOM}")
    
    for pct in pbar:
        target = int(len(pool_copy) * (pct / 100.0))
        labeled.extend(pool_copy[len(labeled):target])
        train_loader = DataLoader(ALDataset(files_A + labeled, processor), batch_size=32, shuffle=True)
        model = train_standard(model, train_loader, desc_prefix=f"Random {pct}%")
        model.save_pretrained(round_dir / f"model_{pct}pct")

# --- 2. MC DROPOUT (CONTINU) ---
print("\n--- [2/5] MC DROPOUT (CONTINU) ---")
strat_dir = output_dir / "MC_Dropout"
os.makedirs(strat_dir, exist_ok=True)
pool_mc, labeled_mc = files_B_pool.copy(), []

model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)

for pct in tqdm(BUDGET_STEPS, desc="MC Dropout Prog."):
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_mc)
    
    model.eval()
    for m in model.modules():
        if 'Dropout' in m.__class__.__name__: m.train()
    loader = DataLoader(ALDataset(pool_mc, processor), batch_size=16)
    uncerts = []
    with torch.no_grad():
        for pix, _ in tqdm(loader, desc=f"   ↳ Scan {pct}%", leave=False):
            preds = np.array([torch.sigmoid(model(pix.to(device)).logits).cpu().numpy() for _ in range(10)])
            uncerts.extend(np.var(preds, axis=0).mean(axis=1))
            
    top_idx = np.argsort(uncerts)[::-1][:needed]
    labeled_mc.extend([pool_mc[i] for i in top_idx])
    pool_mc = [pool_mc[i] for i in range(len(pool_mc)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_mc, processor), batch_size=32, shuffle=True)
    model = train_standard(model, train_loader, desc_prefix=f"Train {pct}%")
    model.save_pretrained(strat_dir / f"model_{pct}pct")

# --- 3. LEARNING LOSS (CONTINU) ---
print("\n--- [3/5] LEARNING LOSS (CONTINU) ---")
strat_dir = output_dir / "LearningLoss"
os.makedirs(strat_dir, exist_ok=True)
pool_ll, labeled_ll = files_B_pool.copy(), []

model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
loss_module = LossModule().to(device)

for pct in tqdm(BUDGET_STEPS, desc="Learning Loss Prog."):
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_ll)
    
    model.eval()
    loss_module.eval()
    loader = DataLoader(ALDataset(pool_ll, processor), batch_size=32)
    preds_loss = []
    with torch.no_grad():
        for pix, _ in tqdm(loader, desc=f"   ↳ Pred. Erreurs {pct}%", leave=False):
            feats = model.resnet(pix.to(device)).pooler_output.squeeze()
            preds_loss.extend(loss_module(feats).cpu().numpy().flatten())
            
    top_idx = np.argsort(preds_loss)[::-1][:needed]
    labeled_ll.extend([pool_ll[i] for i in top_idx])
    pool_ll = [pool_ll[i] for i in range(len(pool_ll)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_ll, processor), batch_size=32, shuffle=True)
    
    model.train()
    loss_module.train()
    optimizer = optim.AdamW([{'params': model.parameters()}, {'params': loss_module.parameters()}], lr=AL_LR)
    crit_mod = nn.MSELoss()
    
    for epoch in range(AL_EPOCHS):
        loop = tqdm(train_loader, desc=f"   ↳ Train {pct}% [Ep {epoch+1}/{AL_EPOCHS}]", leave=False)
        for pix, tgt in loop:
            pix, tgt = pix.to(device), tgt.to(device)
            optimizer.zero_grad()
            outs = model.resnet(pix)
            preds = torch.sigmoid(model.classifier(outs.pooler_output.flatten(1)))
            t_loss = torch.mean((preds - tgt)**2, dim=1)
            p_loss = loss_module(outs.pooler_output.squeeze()).view(-1)
            total_loss = t_loss.mean() + crit_mod(p_loss, t_loss.detach())
            total_loss.backward()
            optimizer.step()
            loop.set_postfix(loss=f"{total_loss.item():.4f}")
            
    model.save_pretrained(strat_dir / f"model_{pct}pct")

# --- 4. K-MEANS (CONTINU) ---
print("\n--- [4/5] K-MEANS (CONTINU) ---")
strat_dir = output_dir / "KMeans"
os.makedirs(strat_dir, exist_ok=True)
pool_km, labeled_km = files_B_pool.copy(), []

def get_feats(current_model, files, desc):
    current_model.eval()
    loader = DataLoader(ALDataset(files, processor), batch_size=64)
    feats = []
    with torch.no_grad():
        for pix, _ in tqdm(loader, desc=f"   ↳ Feats {desc}", leave=False): 
            feats.append(current_model.resnet(pix.to(device)).pooler_output.squeeze().cpu().numpy())
    return np.concatenate(feats) if feats else np.array([])

model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)

for pct in tqdm(BUDGET_STEPS, desc="K-Means Prog."):
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_km)
    
    feats = get_feats(model, pool_km, f"{pct}%")
    kmeans = KMeans(n_clusters=needed, random_state=42, n_init=10).fit(feats)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats)
    top_idx = list(np.unique(closest))
    while len(top_idx) < needed:
        remain = [i for i in range(len(pool_km)) if i not in top_idx]
        top_idx.append(remain[0])
        
    labeled_km.extend([pool_km[i] for i in top_idx])
    pool_km = [pool_km[i] for i in range(len(pool_km)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_km, processor), batch_size=32, shuffle=True)
    model = train_standard(model, train_loader, desc_prefix=f"Train {pct}%")
    model.save_pretrained(strat_dir / f"model_{pct}pct")

# --- 5. OUTLIERS (CONTINU) ---
print("\n--- [5/5] OUTLIERS (CONTINU) ---")
strat_dir = output_dir / "Outliers"
os.makedirs(strat_dir, exist_ok=True)
pool_out, labeled_out = files_B_pool.copy(), []

model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)

for pct in tqdm(BUDGET_STEPS, desc="Outliers Prog."):
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_out)
    
    pool_f = get_feats(model, pool_out, f"Pool {pct}%")
    known_f = get_feats(model, files_A + labeled_out, f"Connus {pct}%")
    
    dists = pairwise_distances(pool_f, known_f)
    min_dists = np.min(dists, axis=1)
    top_idx = []
    for _ in range(needed):
        idx = np.argmax(min_dists)
        top_idx.append(idx)
        min_dists = np.minimum(min_dists, pairwise_distances(pool_f, pool_f[idx].reshape(1, -1)).flatten())
        min_dists[idx] = -1.0
        
    labeled_out.extend([pool_out[i] for i in top_idx])
    pool_out = [pool_out[i] for i in range(len(pool_out)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_out, processor), batch_size=32, shuffle=True)
    model = train_standard(model, train_loader, desc_prefix=f"Train {pct}%")
    model.save_pretrained(strat_dir / f"model_{pct}pct")

print("\n✅ ENTRAÎNEMENT CONTINU TERMINÉ ! Les modèles sont sauvegardés dans AL_Results_Continuous.")