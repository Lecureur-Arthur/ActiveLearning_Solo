import os
# --- FIX POUR L'ERREUR OMP ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ResNetForImageClassification
import time
import datetime # Pour formater le temps joliment
import numpy as np

# --- IMPORTATION SCIPY, MATPLOTLIB & TQDM ---
from scipy.interpolate import make_interp_spline
import matplotlib.pyplot as plt
from tqdm import tqdm # La barre de progression

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 

train_dir = project_root / "data" / "Split" / "Domain_A" / "train"
test_dir = project_root / "data" / "Split" / "Domain_A" / "test"
output_dir = script_dir / "saved_models"

print(f"--- Configuration ---")
print(f"Racine   : {project_root}")
print(f"Train    : {train_dir}")
print(f"Device   : {device}")

if not train_dir.exists():
    raise FileNotFoundError("Dossier Train introuvable.")

train_paths = [str(f) for f in train_dir.glob("*.jpg")]
test_paths = [str(f) for f in test_dir.glob("*.jpg")]

# ==========================================
# 2. DATASET
# ==========================================
class ImageRegressionDataset(Dataset):
    def __init__(self, image_paths, processor):
        self.image_paths = image_paths
        self.processor = processor
        self.max_x = 1920.0 
        self.max_y = 1080.0

    def __len__(self):
        return len(self.image_paths)

    def extract_coords(self, filename):
        name_no_ext = filename.replace('.jpg', '')
        parts = name_no_ext.split('_')
        try:
            x = float(parts[-2])
            y = float(parts[-1])
            return x / self.max_x, y / self.max_y 
        except:
            return 0.0, 0.0

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        target = torch.tensor(self.extract_coords(os.path.basename(path)), dtype=torch.float32)
        
        inputs = self.processor(image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return pixel_values, target

# ==========================================
# 3. PRÉPARATION
# ==========================================
model_name = "microsoft/resnet-18"
print(f"Chargement du modèle {model_name}...")
processor = AutoImageProcessor.from_pretrained(model_name)
model = ResNetForImageClassification.from_pretrained(
    model_name, num_labels=2, ignore_mismatched_sizes=True
).to(device)

train_loader = DataLoader(ImageRegressionDataset(train_paths, processor), batch_size=10, shuffle=True)
test_loader = DataLoader(ImageRegressionDataset(test_paths, processor), batch_size=10, shuffle=False)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss() 

# ==========================================
# 4. ENTRAÎNEMENT (AVEC BARRE DE PROGRESSION)
# ==========================================
epochs = 20
history = {'train_loss': [], 'val_loss': []}

print(f"\nDébut de l'entraînement sur {len(train_paths)} images pour {epochs} époques.")
global_start_time = time.time()

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    
    # --- TQDM : Barre de progression pour l'époque ---
    # On "emballe" le loader dans tqdm
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    
    for pixels, targets in loop:
        pixels, targets = pixels.to(device), targets.to(device)
        
        optimizer.zero_grad()
        preds = torch.sigmoid(model(pixels).logits)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        # Affiche la perte en temps réel à côté de la barre
        loop.set_postfix(loss=loss.item())
    
    avg_train = train_loss / len(train_loader)
    
    # --- Validation ---
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for pixels, targets in test_loader:
            pixels, targets = pixels.to(device), targets.to(device)
            preds = torch.sigmoid(model(pixels).logits)
            val_loss += criterion(preds, targets).item()
            
    avg_val = val_loss / len(test_loader)
    history['train_loss'].append(avg_train)
    history['val_loss'].append(avg_val)
    
    # --- Calcul du temps restant global ---
    elapsed_time = time.time() - global_start_time
    avg_time_per_epoch = elapsed_time / (epoch + 1)
    remaining_epochs = epochs - (epoch + 1)
    est_remaining_time = remaining_epochs * avg_time_per_epoch
    
    # Conversion en format lisible (HH:MM:SS)
    str_remaining = str(datetime.timedelta(seconds=int(est_remaining_time)))
    
    # On utilise print normal ici pour garder une trace dans le terminal après la barre
    print(f" -> Moyenne Train: {avg_train:.4f} | Moyenne Val: {avg_val:.4f} | Fin estimée dans : {str_remaining}")

total_duration = str(datetime.timedelta(seconds=int(time.time() - global_start_time)))
print(f"\n--- Entraînement terminé en {total_duration} ---")

# Sauvegarde
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
processor.save_pretrained(output_dir)
print(f"Modèle sauvegardé : {output_dir}")

# ==========================================
# 5. GRAPHIQUE
# ==========================================
def plot_smooth_curve(data, label, color):
    x = np.array(range(1, len(data) + 1))
    y = np.array(data)
    if len(x) > 3:
        x_smooth = np.linspace(x.min(), x.max(), 300)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        plt.plot(x_smooth, y_smooth, color=color, linewidth=2, label=f'{label} (Lissé)')
        plt.plot(x, y, color=color, alpha=0.3, marker='o', linestyle='None') 
    else:
        plt.plot(x, y, color=color, label=label)

plt.figure(figsize=(10, 6))
plot_smooth_curve(history['train_loss'], 'Train Loss', 'blue')
plot_smooth_curve(history['val_loss'], 'Val Loss', 'orange')
plt.title("Courbe d'apprentissage")
plt.xlabel("Époques")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

graph_path = output_dir / "training_curve.png"
plt.savefig(graph_path)
print(f"Graphique généré : {graph_path}")