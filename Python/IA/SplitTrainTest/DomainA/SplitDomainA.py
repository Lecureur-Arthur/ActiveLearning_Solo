import os
import shutil
import random

def split_domain_a():
    # 1. Récupération du chemin du script actuel
    # Chemin : .../Python/IA/SplitTrainTest/DomainA/SplitDomainA.py
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    
    # 2. Calcul de la racine du projet
    # On remonte 4 niveaux pour atteindre la racine (là où sont 'data', 'README.md', etc.)
    # DomainA (1) -> SplitTrainTest (2) -> IA (3) -> Python (4) -> Racine
    project_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
    
    print(f"Racine du projet détectée : {project_root}")

    # 3. Définition des chemins source et destination
    # Source : data/processed_frames/Domain_A
    source_dir = os.path.join(project_root, 'data', 'processed_frames', 'Domain_A')
    
    # Destination : data/Split/Domain_A (D'après votre structure)
    dest_root = os.path.join(project_root, 'data', 'Split', 'Domain_A')
    train_dir = os.path.join(dest_root, 'train')
    test_dir = os.path.join(dest_root, 'test')

    # Vérification
    if not os.path.exists(source_dir):
        print(f"ERREUR CRITIQUE : Le dossier source est introuvable ici :\n{source_dir}")
        return

    # 4. Création des dossiers cibles
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 5. Récupération des images
    all_images = []
    print(f"Recherche des images dans {source_dir}...")
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_images.append(os.path.join(root, file))

    total_images = len(all_images)
    if total_images == 0:
        print("Aucune image trouvée. Vérifiez que vos dossiers contiennent bien des fichiers .jpg")
        return

    # 6. Mélange et split
    random.shuffle(all_images)
    split_idx = int(total_images * 0.75) # 75% Train
    
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    print(f"Total images : {total_images}")
    print(f" -> Train : {len(train_images)}")
    print(f" -> Test  : {len(test_images)}")

    # 7. Fonction de copie
    def copy_files(file_list, destination):
        count = 0
        for f in file_list:
            try:
                shutil.copy2(f, os.path.join(destination, os.path.basename(f)))
                count += 1
            except Exception as e:
                print(f"Erreur sur {f}: {e}")
        return count

    print("Copie des fichiers Train...")
    c_train = copy_files(train_images, train_dir)
    
    print("Copie des fichiers Test...")
    c_test = copy_files(test_images, test_dir)

    print(f"Terminé ! {c_train + c_test} fichiers copiés dans {dest_root}")

if __name__ == "__main__":
    split_domain_a()