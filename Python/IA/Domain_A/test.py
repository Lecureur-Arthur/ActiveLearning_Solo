import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
import math
from tqdm import tqdm  # Pour la barre de progression

# ==========================================
# 1. CONFIGURATION
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parents[2] 
model_path = script_dir / "saved_models"
test_dir = project_root / "data" / "Split" / "Domain_A" / "test"

# Dimensions Full HD
MAX_X = 1920.0
MAX_Y = 1080.0

print(f"--- TEST AVEC MSE GLOBALE ---")
print(f"Modèle : {model_path}")

if not model_path.exists():
    raise FileNotFoundError("Modèle introuvable.")

# ==========================================
# 2. CHARGEMENT
# ==========================================
try:
    processor = AutoImageProcessor.from_pretrained(model_path)
    model = ResNetForImageClassification.from_pretrained(model_path).to(device)
    model.eval()
except Exception as e:
    print(f"ERREUR CHARGEMENT : {e}")
    exit()

criterion = nn.MSELoss()

# ==========================================
# 3. FONCTIONS
# ==========================================
def extract_ground_truth(filename):
    name_no_ext = filename.rsplit('.', 1)[0]
    parts = name_no_ext.split('_')
    try:
        x = float(parts[-2])
        y = float(parts[-1])
        return x, y
    except Exception as e:
        return None, None

def predict_and_loss(image_path, true_x, true_y):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs).logits
        probs = torch.sigmoid(outputs).cpu().squeeze()
        
        # Prédiction
        pred_x = probs[0].item() * MAX_X
        pred_y = probs[1].item() * MAX_Y
        
        mse_loss = None
        if true_x is not None:
            # Normalisation pour la Loss (0.0 à 1.0)
            target = torch.tensor([true_x / MAX_X, true_y / MAX_Y], dtype=torch.float32)
            mse_loss = criterion(probs, target).item()
            
    return pred_x, pred_y, mse_loss

# ==========================================
# 4. CALCUL DE LA MSE GLOBALE (SUR TOUT LE TEST)
# ==========================================
image_files = list(test_dir.glob("*.jpg"))
if not image_files:
    print("Pas d'images.")
    exit()

print(f"\nCalcul de la MSE sur les {len(image_files)} images de test...")
total_mse = 0.0
valid_images_count = 0

# On parcourt TOUTES les images pour calculer la moyenne
for img_path in tqdm(image_files, desc="Evaluation"):
    filename = os.path.basename(img_path)
    true_x, true_y = extract_ground_truth(filename)
    
    # On ne calcule l'erreur que si on connait la vraie position
    if true_x is not None:
        _, _, loss = predict_and_loss(img_path, true_x, true_y)
        total_mse += loss
        valid_images_count += 1

# Moyenne
global_mse = total_mse / valid_images_count if valid_images_count > 0 else 0.0

print("\n" + "="*40)
print(f" RÉSULTAT FINAL SUR {valid_images_count} IMAGES")
print("="*40)
print(f"Global MSE Loss : {global_mse:.6f}")
print("="*40 + "\n")

# ==========================================
# 5. AFFICHAGE VISUEL (3 SAMPLES)
# ==========================================
samples = random.sample(image_files, min(3, len(image_files)))

plt.figure(figsize=(16, 7)) # Un peu plus haut pour le titre global

# Ajout du titre global avec la MSE calculée
plt.suptitle(f"Global Test Set MSE: {global_mse:.6f} (sur {valid_images_count} images)", fontsize=16, fontweight='bold')

for i, img_path in enumerate(samples):
    filename = os.path.basename(img_path)
    true_x, true_y = extract_ground_truth(filename)
    pred_x, pred_y, loss_val = predict_and_loss(img_path, true_x, true_y)
    
    ax = plt.subplot(1, 3, i+1)
    ax.set_xlim(0, MAX_X)
    ax.set_ylim(MAX_Y, 0) # Inversion Y
    ax.set_aspect('equal')
    
    rect = patches.Rectangle((0, 0), MAX_X, MAX_Y, linewidth=2, edgecolor='#333', facecolor='#f0f0f0')
    ax.add_patch(rect)
    
    title_text = f"Image {i+1}\n"
    
    plt.scatter(pred_x, pred_y, c='red', s=150, marker='x', label='IA', linewidths=3, zorder=5)
    
    if true_x is not None:
        plt.scatter(true_x, true_y, c='lime', s=150, marker='o', label='Réel', edgecolors='black', zorder=4)
        plt.plot([true_x, pred_x], [true_y, pred_y], 'k--', alpha=0.5, label='Distance')
        dist_px = math.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        # On affiche la Loss spécifique de CETTE image
        title_text += f"Dist: {dist_px:.0f}px | MSE: {loss_val:.5f}"
    else:
        plt.text(MAX_X/2, MAX_Y/2, "COORDONNÉES\nINCONNUES", ha='center', color='red')
        title_text += "Nom invalide"

    plt.title(title_text, fontsize=10)
    if i == 0: plt.legend(loc='upper right')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Laisse de la place pour le suptitle
plt.show()