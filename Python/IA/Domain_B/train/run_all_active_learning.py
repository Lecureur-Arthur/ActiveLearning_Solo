import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt
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
# Remonte de 3 crans : train -> Domain_B -> IA -> Python -> Racine
project_root = script_dir.parents[3] 

# --- CHEMINS ---
path_train_A = project_root / "data" / "Split" / "Domain_A" / "train"
path_train_B = project_root / "data" / "Split" / "Domain_B" / "train"
path_test_B  = project_root / "data" / "Split" / "Domain_B" / "test"
model_A_path = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Sauvegarde dans Domain_B (Parent de "train")
output_dir = script_dir.parent / "AL_Results_Automated"
os.makedirs(output_dir, exist_ok=True)

# --- PARAMETRES ---
MAX_X, MAX_Y = 1920.0, 1080.0
BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS
AL_EPOCHS = 5      
AL_LR = 1e-5 
N_ROUNDS_RANDOM = 3

print("="*60)
print("🚀 LANCEMENT DU PIPELINE ACTIVE LEARNING 🚀")
print(f"🖥️  Appareil utilisé : {device}")
print("="*60)

# ==========================================
# 2. DATASET ET UTILITAIRES DE BASE
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
        try:
            x = float(name.split('_')[-2]) / MAX_X
            y = float(name.split('_')[-1]) / MAX_Y
        except: x, y = 0.0, 0.0
        target = torch.tensor([x, y], dtype=torch.float32)
        return self.processor(image, return_tensors="pt")['pixel_values'].squeeze(0), target

def evaluate_mse(model, loader):
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for pix, tgt in loader:
            out = torch.sigmoid(model(pix.to(device)).logits)
            total_loss += criterion(out, tgt.to(device)).item()
    return total_loss / len(loader)

def train_standard(model, train_loader, desc_prefix="Train"):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=AL_LR)
    criterion = nn.MSELoss()
    
    for epoch in range(AL_EPOCHS):
        # Barre de progression pour chaque époque d'entraînement
        loop = tqdm(train_loader, desc=f"   ↳ {desc_prefix} [Ep {epoch+1}/{AL_EPOCHS}]", leave=False)
        for pix, tgt in loop:
            optimizer.zero_grad()
            loss = criterion(torch.sigmoid(model(pix.to(device)).logits), tgt.to(device))
            loss.backward()
            optimizer.step()
            # Affichage de la perte en temps réel à droite de la barre
            loop.set_postfix(loss=f"{loss.item():.4f}")
    return model

def plot_vs_random(strat_name, strat_scores, rand_means, rand_stds):
    plt.figure(figsize=(10, 6))
    plt.plot(BUDGET_WITH_ZERO, rand_means, color='gray', linestyle='--', label='Random (Moyenne)')
    plt.fill_between(BUDGET_WITH_ZERO, rand_means - rand_stds, rand_means + rand_stds, color='gray', alpha=0.2)
    plt.plot(BUDGET_WITH_ZERO, strat_scores, marker='o', linewidth=2.5, label=strat_name)
    plt.title(f"{strat_name} vs Random Baseline")
    plt.xlabel("% Domain B")
    plt.ylabel("MSE Loss")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.xticks(BUDGET_WITH_ZERO)
    plt.savefig(output_dir / f"Compare_{strat_name.replace(' ', '_')}.png")
    plt.close()

# ==========================================
# 3. PREPARATION DES DONNEES
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
files_B_test = list(path_test_B.glob("**/*.jpg"))
processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

global_results = {}

# ==========================================
# 4. PHASE 1 : RANDOM ROBUST
# ==========================================
print("\n--- [1/5] RANDOM ROBUST BASELINE ---")

# Création du dossier principal pour le Random
strat_dir_random = output_dir / "Random"
os.makedirs(strat_dir_random, exist_ok=True)

random_results = np.zeros((N_ROUNDS_RANDOM, len(BUDGET_WITH_ZERO)))

for r in range(N_ROUNDS_RANDOM):
    random.seed(42 + r)
    pool_copy = files_B_pool.copy()
    random.shuffle(pool_copy)
    labeled = []
    
    # Création d'un sous-dossier pour chaque round (Round_1, Round_2...)
    round_dir = strat_dir_random / f"Round_{r+1}"
    os.makedirs(round_dir, exist_ok=True)
    
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    random_results[r, 0] = evaluate_mse(model, test_loader)
    
    pbar = tqdm(BUDGET_STEPS, desc=f"Round Random {r+1}/{N_ROUNDS_RANDOM}")
    for step_idx, pct in enumerate(pbar):
        target = int(len(pool_copy) * (pct / 100.0))
        labeled.extend(pool_copy[len(labeled):target])
        
        train_loader = DataLoader(ALDataset(files_A + labeled, processor), batch_size=32, shuffle=True)
        model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
        model = train_standard(model, train_loader, desc_prefix=f"Random {pct}%")
        
        mse = evaluate_mse(model, test_loader)
        random_results[r, step_idx+1] = mse
        
        # --- NOUVEAU : SAUVEGARDE DU MODELE RANDOM ---
        save_path = round_dir / f"model_{pct}pct"
        model.save_pretrained(save_path)
        processor.save_pretrained(save_path)
        # ---------------------------------------------

rand_mean = np.mean(random_results, axis=0)
rand_std = np.std(random_results, axis=0)
global_results["Random"] = rand_mean

# ==========================================
# 5. PHASE 2 : MC DROPOUT
# ==========================================
print("\n--- [2/5] MC DROPOUT ---")
strat_dir = output_dir / "MC_Dropout"
os.makedirs(strat_dir, exist_ok=True)
scores_mc = [rand_mean[0]]
pool_mc, labeled_mc = files_B_pool.copy(), []

pbar_mc = tqdm(BUDGET_STEPS, desc="MC Dropout Prog.")
for pct in pbar_mc:
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_mc)
    
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model.eval()
    for m in model.modules():
        if 'Dropout' in m.__class__.__name__: m.train()
            
    loader = DataLoader(ALDataset(pool_mc, processor), batch_size=16)
    uncerts = []
    
    scan_loop = tqdm(loader, desc=f"   ↳ Scan Incertitude {pct}%", leave=False)
    with torch.no_grad():
        for pix, _ in scan_loop:
            preds = np.array([torch.sigmoid(model(pix.to(device)).logits).cpu().numpy() for _ in range(10)])
            uncerts.extend(np.var(preds, axis=0).mean(axis=1))
            
    top_idx = np.argsort(uncerts)[::-1][:needed]
    labeled_mc.extend([pool_mc[i] for i in top_idx])
    pool_mc = [pool_mc[i] for i in range(len(pool_mc)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_mc, processor), batch_size=32, shuffle=True)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model = train_standard(model, train_loader, desc_prefix=f"Train {pct}%")
    
    mse = evaluate_mse(model, test_loader)
    scores_mc.append(mse)
    model.save_pretrained(strat_dir / f"model_{pct}pct")
    tqdm.write(f"✅ {pct}% terminé -> MSE: {mse:.6f}")

global_results["MC Dropout"] = scores_mc
plot_vs_random("MC Dropout", scores_mc, rand_mean, rand_std)

# ==========================================
# 6. PHASE 3 : LEARNING LOSS
# ==========================================
print("\n--- [3/5] LEARNING LOSS ---")
strat_dir = output_dir / "LearningLoss"
os.makedirs(strat_dir, exist_ok=True)
scores_ll = [rand_mean[0]]
pool_ll, labeled_ll = files_B_pool.copy(), []

class LossModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x): return self.net(x)

pbar_ll = tqdm(BUDGET_STEPS, desc="Learning Loss Prog.")
for pct in pbar_ll:
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_ll)
    
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    loss_module = LossModule().to(device)
    model.eval(); loss_module.eval()
    
    loader = DataLoader(ALDataset(pool_ll, processor), batch_size=32)
    preds_loss = []
    
    scan_loop = tqdm(loader, desc=f"   ↳ Pred. Erreurs {pct}%", leave=False)
    with torch.no_grad():
        for pix, _ in scan_loop:
            feats = model.resnet(pix.to(device)).pooler_output.squeeze()
            preds_loss.extend(loss_module(feats).cpu().numpy().flatten())
            
    top_idx = np.argsort(preds_loss)[::-1][:needed]
    labeled_ll.extend([pool_ll[i] for i in top_idx])
    pool_ll = [pool_ll[i] for i in range(len(pool_ll)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_ll, processor), batch_size=32, shuffle=True)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    loss_module = LossModule().to(device)
    model.train(); loss_module.train()
    optimizer = optim.AdamW([{'params': model.parameters()}, {'params': loss_module.parameters()}], lr=AL_LR)
    crit_task, crit_mod = nn.MSELoss(reduction='none'), nn.MSELoss()
    
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
            
    mse = evaluate_mse(model, test_loader)
    scores_ll.append(mse)
    model.save_pretrained(strat_dir / f"model_{pct}pct")
    tqdm.write(f"✅ {pct}% terminé -> MSE: {mse:.6f}")

global_results["Learning Loss"] = scores_ll
plot_vs_random("Learning Loss", scores_ll, rand_mean, rand_std)

# ==========================================
# 7. PHASE 4 : K-MEANS
# ==========================================
print("\n--- [4/5] K-MEANS ---")
strat_dir = output_dir / "KMeans"
os.makedirs(strat_dir, exist_ok=True)
scores_km = [rand_mean[0]]
pool_km, labeled_km = files_B_pool.copy(), []

def get_feats(model, files, desc):
    model.eval()
    loader = DataLoader(ALDataset(files, processor), batch_size=64)
    feats = []
    loop = tqdm(loader, desc=f"   ↳ Extr. Features {desc}", leave=False)
    with torch.no_grad():
        for pix, _ in loop: 
            feats.append(model.resnet(pix.to(device)).pooler_output.squeeze().cpu().numpy())
    return np.concatenate(feats) if feats else np.array([])

pbar_km = tqdm(BUDGET_STEPS, desc="K-Means Prog.")
for pct in pbar_km:
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_km)
    
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    feats = get_feats(model, pool_km, f"{pct}%")
    
    tqdm.write(f"   ↳ K-Means Clustering en cours...")
    kmeans = KMeans(n_clusters=needed, random_state=42, n_init=10).fit(feats)
    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, feats)
    
    top_idx = list(np.unique(closest))
    while len(top_idx) < needed:
        remain = [i for i in range(len(pool_km)) if i not in top_idx]
        top_idx.append(remain[0])
        
    labeled_km.extend([pool_km[i] for i in top_idx])
    pool_km = [pool_km[i] for i in range(len(pool_km)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_km, processor), batch_size=32, shuffle=True)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model = train_standard(model, train_loader, desc_prefix=f"Train {pct}%")
    
    mse = evaluate_mse(model, test_loader)
    scores_km.append(mse)
    model.save_pretrained(strat_dir / f"model_{pct}pct")
    tqdm.write(f"✅ {pct}% terminé -> MSE: {mse:.6f}")

global_results["K-Means"] = scores_km
plot_vs_random("K-Means", scores_km, rand_mean, rand_std)

# ==========================================
# 8. PHASE 5 : OUTLIERS
# ==========================================
print("\n--- [5/5] OUTLIERS ---")
strat_dir = output_dir / "Outliers"
os.makedirs(strat_dir, exist_ok=True)
scores_out = [rand_mean[0]]
pool_out, labeled_out = files_B_pool.copy(), []

pbar_out = tqdm(BUDGET_STEPS, desc="Outliers Prog.")
for pct in pbar_out:
    target = int(len(files_B_pool) * (pct / 100.0))
    needed = target - len(labeled_out)
    
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    pool_f = get_feats(model, pool_out, f"Pool {pct}%")
    known_f = get_feats(model, files_A + labeled_out, f"Connus {pct}%")
    
    tqdm.write(f"   ↳ Calcul des distances...")
    dists = pairwise_distances(pool_f, known_f)
    min_dists = np.min(dists, axis=1)
    
    top_idx = []
    for _ in range(needed):
        idx = np.argmax(min_dists)
        top_idx.append(idx)
        new_f = pool_f[idx].reshape(1, -1)
        dists_to_new = pairwise_distances(pool_f, new_f).flatten()
        min_dists = np.minimum(min_dists, dists_to_new)
        min_dists[idx] = -1.0
        
    labeled_out.extend([pool_out[i] for i in top_idx])
    pool_out = [pool_out[i] for i in range(len(pool_out)) if i not in top_idx]
    
    train_loader = DataLoader(ALDataset(files_A + labeled_out, processor), batch_size=32, shuffle=True)
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model = train_standard(model, train_loader, desc_prefix=f"Train {pct}%")
    
    mse = evaluate_mse(model, test_loader)
    scores_out.append(mse)
    model.save_pretrained(strat_dir / f"model_{pct}pct")
    tqdm.write(f"✅ {pct}% terminé -> MSE: {mse:.6f}")

global_results["Outliers"] = scores_out
plot_vs_random("Outliers", scores_out, rand_mean, rand_std)

# ==========================================
# 9. GRAPH FINAL GLOBAL
# ==========================================
print("\n--- GENERATION DU GRAPHIQUE FINAL ---")
plt.figure(figsize=(12, 8))
plt.plot(BUDGET_WITH_ZERO, rand_mean, color='gray', linestyle='--', linewidth=2, label='Random (Moy)')
plt.fill_between(BUDGET_WITH_ZERO, rand_mean - rand_std, rand_mean + rand_std, color='gray', alpha=0.15)

colors = {"MC Dropout": "purple", "Learning Loss": "orange", "K-Means": "green", "Outliers": "red"}
markers = {"MC Dropout": "s", "Learning Loss": "^", "K-Means": "d", "Outliers": "x"}

for name, scores in global_results.items():
    if name == "Random": continue
    plt.plot(BUDGET_WITH_ZERO, scores, marker=markers[name], color=colors[name], linewidth=2.5, label=name)

plt.title("Comparaison Finale des Stratégies d'Active Learning")
plt.xlabel("% Domain B Annoté")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.xticks(BUDGET_WITH_ZERO)
plt.savefig(output_dir / "GRAND_FINAL_BENCHMARK.png")
print(f"\n🎉 TOUT EST TERMINE ! Resultats dans : {output_dir}")