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

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

# Dossier de Test B
path_test_B = project_root / "data" / "Split" / "Domain_B" / "test"

# Chemins des Checkpoints (Modèles sauvegardés)
path_random = script_dir / "AL_Results" / "Random_Strategy" / "checkpoints"
path_mcdrop = script_dir / "AL_Results" / "Uncertainty_MC_Dropout" / "checkpoints"

# Modèle de base (0%)
path_base_model = project_root / "Python" / "IA" / "Domain_A" / "saved_models"

# Dossier de sortie
output_dir = script_dir / "AL_Results" / "Comparison"
os.makedirs(output_dir, exist_ok=True)

# Les étapes enregistrées
STEPS = [0, 1, 2, 5, 10, 20, 50]
MAX_X = 1920.0
MAX_Y = 1080.0

print("--- COMPARATEUR DE STRATÉGIES ---")
print(f"Random Checkpoints : {path_random}")
print(f"MC Drop Checkpoints: {path_mcdrop}")

# ==========================================
# 2. DATASET & OUTILS
# ==========================================
class CompDataset(Dataset):
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

def get_mse(model_path, loader):
    """Charge un modèle et calcule sa MSE"""
    if not model_path.exists():
        return None # Modèle manquant
        
    try:
        model = ResNetForImageClassification.from_pretrained(model_path).to(device)
        model.eval()
        criterion = nn.MSELoss()
        total_loss = 0.0
        with torch.no_grad():
            for pix, tgt in loader:
                pix, tgt = pix.to(device), tgt.to(device)
                out = torch.sigmoid(model(pix).logits)
                loss = criterion(out, tgt)
                total_loss += loss.item()
        return total_loss / len(loader)
    except Exception as e:
        print(f"Erreur sur {model_path.name}: {e}")
        return None

# ==========================================
# 3. PRÉPARATION
# ==========================================
files_B = list(path_test_B.glob("**/*.jpg"))
if not files_B: files_B = list(path_test_B.glob("*.jpg"))

print(f"Évaluation sur {len(files_B)} images de test.")

# Processeur par défaut
processor = AutoImageProcessor.from_pretrained(path_base_model)
loader = DataLoader(CompDataset(files_B, processor), batch_size=32, shuffle=False)

# ==========================================
# 4. BOUCLE D'ÉVALUATION
# ==========================================
mse_random = []
mse_mcdrop = []

print("\n>>> Calcul des scores MSE...")

for pct in tqdm(STEPS, desc="Steps"):
    # 1. Cas 0% (Commun aux deux)
    if pct == 0:
        score = get_mse(path_base_model, loader)
        mse_random.append(score)
        mse_mcdrop.append(score)
        continue

    # 2. Random Strategy
    # Nom du dossier : model_random_5percent
    folder_rand = path_random / f"model_random_{pct}percent"
    score_rand = get_mse(folder_rand, loader)
    mse_random.append(score_rand)

    # 3. MC Dropout Strategy
    # Nom du dossier : model_mcdropout_5percent
    folder_mc = path_mcdrop / f"model_mcdropout_{pct}percent"
    score_mc = get_mse(folder_mc, loader)
    mse_mcdrop.append(score_mc)

# ==========================================
# 5. AFFICHAGE DES RÉSULTATS
# ==========================================
print("\n--- RÉSULTATS COMPARATIFS ---")
print(f"{'PCT':<5} | {'RANDOM':<10} | {'MC DROP':<10} | {'GAIN':<10}")
print("-" * 45)

valid_steps = []
clean_rand = []
clean_mc = []

for i, pct in enumerate(STEPS):
    r = mse_random[i]
    m = mse_mcdrop[i]
    
    if r is not None and m is not None:
        gain = ((r - m) / r) * 100 # Positif si MC est meilleur (Loss plus basse)
        print(f"{pct:<5} | {r:.6f}   | {m:.6f}   | {gain:+.2f}%")
        
        valid_steps.append(pct)
        clean_rand.append(r)
        clean_mc.append(m)
    else:
        print(f"{pct:<5} | {'MISSING':<10} | {'MISSING':<10} | N/A")

# ==========================================
# 6. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))

# Courbe Random (Bleu)
plt.plot(valid_steps, clean_rand, 'o--', color='blue', label='Random Sampling (Hasard)', linewidth=2, alpha=0.7)

# Courbe MC Dropout (Rouge/Violet)
plt.plot(valid_steps, clean_mc, 's-', color='#800080', label='Uncertainty (MC Dropout)', linewidth=3)

plt.title("Active Learning : Comparaison des Stratégies\n(MSE Loss sur Domain B)")
plt.xlabel("% de données Domain B annotées")
plt.ylabel("Erreur MSE (Plus bas = Mieux)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(STEPS)

save_path = output_dir / "comparison_Random_vs_MCDropout.png"
plt.savefig(save_path)
print(f"\nGraphique sauvegardé : {save_path}")
plt.show()