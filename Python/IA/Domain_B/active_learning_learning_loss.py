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
import numpy as np
from tqdm import tqdm
import random

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

results_dir = script_dir / "AL_Results" / "Uncertainty_LearningLoss"
checkpoints_dir = results_dir / "checkpoints"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

MAX_X = 1920.0
MAX_Y = 1080.0

BUDGET_STEPS = [1, 2, 5, 10, 20, 50] 
AL_EPOCHS = 5      
AL_LR = 1e-5
LOSS_MODULE_WEIGHT = 1.0 # Importance de l'apprentissage de la loss

print(f"--- ACTIVE LEARNING (UNCERTAINTY - LEARNING LOSS) ---")

if not model_A_path.exists():
    raise FileNotFoundError("Modèle A introuvable.")

# ==========================================
# 2. LE MODULE DE PRÉDICTION D'ERREUR
# ==========================================
class LossPredictionModule(nn.Module):
    def __init__(self, feature_dim=512):
        super(LossPredictionModule, self).__init__()
        # Un petit réseau (MLP) qui prend les features (512) et sort un scalaire (1)
        # Ce scalaire est l'estimation de l'erreur MSE future
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1) # Sortie : L'erreur prédite
        )

    def forward(self, features):
        return self.net(features)

# ==========================================
# 3. DATASET
# ==========================================
class ALDataset(Dataset):
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
        return inputs['pixel_values'].squeeze(0), target

def evaluate_mse(model, loader):
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
# 4. FONCTION DE SÉLECTION (LOSS PREDICTION)
# ==========================================
def select_samples_learning_loss(model, loss_module, pool_files, n_needed, processor):
    """
    Passe tout le pool dans le Loss Module et prend les images
    ayant la plus grande erreur prédite.
    """
    print(f" -> Estimation des erreurs sur {len(pool_files)} images...")
    
    model.eval()
    loss_module.eval()
    
    predicted_losses = []
    
    pool_loader = DataLoader(ALDataset(pool_files, processor), batch_size=32, shuffle=False)
    
    with torch.no_grad():
        for pix, _ in tqdm(pool_loader, desc="Scanning Pool"):
            pix = pix.to(device)
            
            # 1. On récupère les features du modèle principal
            # HuggingFace ResNet sort 'pooler_output' (Batch, 512, 1, 1)
            outputs = model.resnet(pix)
            features = outputs.pooler_output.squeeze() # (Batch, 512)
            
            # 2. Le petit module devine l'erreur
            pred_loss = loss_module(features) # (Batch, 1)
            
            # On stocke les valeurs
            predicted_losses.extend(pred_loss.cpu().numpy().flatten())
            
    # Conversion
    predicted_losses = np.array(predicted_losses)
    
    # On veut les indices avec la plus GRANDE erreur prédite
    # argsort trie croissant, donc on prend la fin
    sorted_indices = np.argsort(predicted_losses)[::-1]
    top_indices = sorted_indices[:n_needed]
    
    selected_files = [pool_files[i] for i in top_indices]
    
    # Mise à jour du pool
    remaining_pool = [pool_files[i] for i in range(len(pool_files)) if i not in top_indices]
    
    avg_predicted_error = predicted_losses[top_indices].mean()
    
    return selected_files, remaining_pool, avg_predicted_error

# ==========================================
# 5. PRÉPARATION
# ==========================================
files_A = list(path_train_A.glob("*.jpg"))
files_B_pool = list(path_train_B.glob("**/*.jpg"))
if not files_B_pool: files_B_pool = list(path_train_B.glob("*.jpg"))

files_B_test = list(path_test_B.glob("**/*.jpg"))
if not files_B_test: files_B_test = list(path_test_B.glob("*.jpg"))

processor = AutoImageProcessor.from_pretrained(model_A_path)
test_loader = DataLoader(ALDataset(files_B_test, processor), batch_size=32, shuffle=False)

labeled_B = []
results_mse = []
BUDGET_WITH_ZERO = [0] + BUDGET_STEPS

# IMPORTANT : Le Loss Module doit être persistant ou réinitialisé ?
# En général, on le réentraîne à chaque cycle pour qu'il s'adapte au nouveau modèle.
loss_module = LossPredictionModule(feature_dim=512).to(device)

# ==========================================
# 6. BASELINE (0%)
# ==========================================
print("\n--- BASELINE (0%) ---")
model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
initial_mse = evaluate_mse(model, test_loader)
results_mse.append(initial_mse)
print(f" -> MSE Initiale : {initial_mse:.6f}")

# ==========================================
# 7. BOUCLE ACTIVE LEARNING
# ==========================================
current_pool = files_B_pool

for pct in BUDGET_STEPS:
    print(f"\n>>> VISÉE : {pct}% du Domain B (Stratégie: Predict Own Errors)")
    
    # --- PHASE 1 : SÉLECTION ---
    target_total = int(len(files_B_pool) * (pct / 100.0))
    current_count = len(labeled_B)
    n_needed = target_total - current_count
    
    if n_needed > 0:
        # On utilise le modèle et le loss module de l'étape précédente
        new_data, current_pool, avg_err = select_samples_learning_loss(model, loss_module, current_pool, n_needed, processor)
        labeled_B.extend(new_data)
        print(f" -> Ajout de {len(new_data)} images (Erreur prédite moy: {avg_err:.4f})")
    
    # --- PHASE 2 : ENTRAÎNEMENT ---
    # On recharge le modèle A de base
    model = ResNetForImageClassification.from_pretrained(model_A_path).to(device)
    model.train()
    
    # On réinitialise aussi le Loss Module pour qu'il apprenne sur le nouveau dataset mixte
    loss_module = LossPredictionModule(feature_dim=512).to(device)
    loss_module.train()
    
    # Dataset Mixte
    mixed_files = files_A + labeled_B
    train_loader = DataLoader(ALDataset(mixed_files, processor), batch_size=32, shuffle=True)
    
    # Optimiseur : Il doit mettre à jour le Modèle ET le Loss Module
    optimizer = optim.AdamW([
        {'params': model.parameters()},
        {'params': loss_module.parameters()}
    ], lr=AL_LR)
    
    criterion_task = nn.MSELoss(reduction='none') # 'none' pour avoir la loss par image
    criterion_module = nn.MSELoss() # Pour apprendre à prédire la loss
    
    for epoch in range(AL_EPOCHS):
        loop = tqdm(train_loader, desc=f"Train {pct}%", leave=False)
        epoch_loss = 0
        
        for pix, tgt in loop:
            pix, tgt = pix.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            
            # 1. Forward Pass Modèle Principal
            outputs = model.resnet(pix) # On récupère les features interne
            features = outputs.pooler_output.squeeze() # (Batch, 512)
            
            # On passe les features dans le classifier (la tête du ResNet) pour avoir la prédiction
            # Note : ResNetForImageClassification a une couche 'classifier' qui prend 'pooler_output'
            # On doit recréer la logique de sortie manuellement ou utiliser le modèle complet
            # Méthode simple : on utilise le modèle complet pour la prédiction finale
            logits = model.classifier(outputs.pooler_output.flatten(1))
            preds = torch.sigmoid(logits)
            
            # 2. Calcul de la VRAIE Loss (Task Loss)
            # (Batch_size,) : Un chiffre par image
            real_loss_per_image = torch.mean((preds - tgt) ** 2, dim=1) 
            task_loss = real_loss_per_image.mean()
            
            # 3. Forward Pass Loss Module
            # Le module essaie de prédire 'real_loss_per_image'
            pred_loss = loss_module(features) # (Batch, 1)
            pred_loss = pred_loss.view(-1)    # (Batch,)
            
            # 4. Calcul de la Loss du Module
            # On détache real_loss car on ne veut pas modifier le modèle principal 
            # pour qu'il fasse exprès d'avoir une loss facile à prédire.
            module_loss = criterion_module(pred_loss, real_loss_per_image.detach())
            
            # 5. Loss Totale
            total_loss = task_loss + (LOSS_MODULE_WEIGHT * module_loss)
            
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += task_loss.item()
            
    # --- PHASE 3 : EVALUATION ---
    current_mse = evaluate_mse(model, test_loader)
    results_mse.append(current_mse)
    print(f" -> MSE Résultat : {current_mse:.6f}")
    
    # Sauvegarde
    folder_name = f"model_learningloss_{pct}percent"
    save_path = checkpoints_dir / folder_name
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    processor.save_pretrained(save_path)
    # On sauvegarde aussi le petit module (pourrait servir)
    torch.save(loss_module.state_dict(), save_path / "loss_module.pth")

# ==========================================
# 8. GRAPHIQUE
# ==========================================
plt.figure(figsize=(10, 6))
plt.plot(BUDGET_WITH_ZERO, results_mse, '^-', linewidth=2, color='darkorange', label='Learning Loss Strategy')
plt.title("Active Learning Curve (Module de Prédiction d'Erreur)")
plt.xlabel("% de données Domain B")
plt.ylabel("MSE Loss")
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(BUDGET_WITH_ZERO)
plt.legend()

out_file = results_dir / "learning_loss_curve.png"
plt.savefig(out_file)
print(f"\nTerminé ! Graphique : {out_file}")
plt.show()

print("\n--- RÉSUMÉ LEARNING LOSS ---")
for p, m in zip(BUDGET_WITH_ZERO, results_mse):
    print(f"{p}% -> MSE: {m:.6f}")