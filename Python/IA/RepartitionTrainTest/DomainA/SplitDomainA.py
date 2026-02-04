import os
import shutil
import random
from pathlib import Path

def split_domain_a(source_dir, dest_root, train_ratio=0.75):

    # Définition des chemins de destination
    train_dir = os.path.join(dest_root, 'train')
    test_dir = os.path.join(dest_root, 'test')

    # Création des dossiers s'ils n'exsitent pas
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Collecte de toutes les images .jpg dans Domain_A et ses sous-dossiers
    all_images = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endwith('.jpg'):
                all_images.append(os.path.join(root, file))
    
    if not all_images:
        print("Aucune image trouvée dans le répertoire source.")
        return
    
    # Mélange aléatoire des images pour un split équitable
    random.shuffle(all_images)

    # Calcul de l'index de séparation
    split_idx = int(len(all_images) * train_ratio)
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    # Fonction utilitaire pour copier les fichiers
    def copy_files(files, destination):
        for f in files:
            # On garde le nom original du fichier
            shutil.copy2(f, os.path.join(destination, os.path.basename(f)))

    # Exécution de la copie
    print(f"Copie de {len(train_images)} images vers {train_dir}...")
    copy_files(train_images, train_dir)

    print(f"Copie de {len(test_images)} images vers {test_dir}...")
    copy_files(test_images, test_dir)

    print("Opération terminée avec succès.")

if __name__ == "__main__":
    
    # Configuration des chemin selon l'environnement
    src = os.path.join('data', 'processed_frames', 'Domain_A')
    dst = os.path.join('data', 'Split', 'Domain_A')

    split_domain_a(src, dst)