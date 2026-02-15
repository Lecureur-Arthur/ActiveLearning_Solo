import os
import shutil
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION DES CHEMINS
# ==========================================
script_dir = Path(__file__).resolve().parent
# Ajustez le nombre de .parents selon l'emplacement exact de ce script
project_root = script_dir.parents[3] 

source_dir = project_root / "data" / "processed_framed" / "Domain_B"
dest_train = project_root / "data" / "Split" / "Domain_B" / "train"
dest_test  = project_root / "data" / "Split" / "Domain_B" / "test"

# Définition stricte des dossiers de test
TEST_FOLDERS = ["9", "10", "11"]

print("--- NOUVEAU SPLIT DOMAIN B (Par Dossier) ---")
print(f"Source : {source_dir}")

if not source_dir.exists():
    raise FileNotFoundError(f"Le dossier source n'existe pas : {source_dir}")

# ==========================================
# 2. NETTOYAGE DES ANCIENS SPLITS
# ==========================================
print("Nettoyage des anciens dossiers de split...")
if dest_train.exists():
    shutil.rmtree(dest_train)
if dest_test.exists():
    shutil.rmtree(dest_test)

os.makedirs(dest_train, exist_ok=True)
os.makedirs(dest_test, exist_ok=True)

# ==========================================
# 3. RÉPARTITION DES FICHIERS
# ==========================================
# On liste tous les sous-dossiers dans Domain_B
subfolders = [f for f in source_dir.iterdir() if f.is_dir()]

train_count = 0
test_count = 0

for folder in subfolders:
    folder_name = folder.name
    
    # On détermine la destination
    if folder_name in TEST_FOLDERS:
        destination = dest_test
        print(f" -> Dossier '{folder_name}' assigné au TEST")
    else:
        destination = dest_train
        print(f" -> Dossier '{folder_name}' assigné au TRAIN")
        
    # On copie les images
    images = list(folder.glob("*.jpg"))
    for img_path in tqdm(images, desc=f"Copie {folder_name}", leave=False):
        shutil.copy2(img_path, destination / img_path.name)
        
    if folder_name in TEST_FOLDERS:
        test_count += len(images)
    else:
        train_count += len(images)

print("\n--- RÉSUMÉ DU SPLIT ---")
print(f"Images dans TRAIN (Pool) : {train_count}")
print(f"Images dans TEST         : {test_count}")
print("Opération terminée avec succès !")