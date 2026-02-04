import os
import shutil
import random

def split_domain_b():
    # 1. Récupération du chemin du script actuel
    # Chemin attendu : .../Python/IA/SplitTrainTest/DomainB/SplitDomainB.py
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    
    # 2. Calcul de la racine du projet (Remonte 4 niveaux : DomainB -> SplitTrainTest -> IA -> Python -> Racine)
    project_root = os.path.abspath(os.path.join(script_dir, '../../../../'))
    
    print(f"Racine du projet détectée : {project_root}")

    # 3. Définition des chemins
    # Source : data/processed_frames/Domain_B
    source_dir = os.path.join(project_root, 'data', 'processed_frames', 'Domain_B')
    
    # Destination : data/Split/Domain_B
    dest_root = os.path.join(project_root, 'data', 'Split', 'Domain_B')
    train_dir = os.path.join(dest_root, 'train')
    test_dir = os.path.join(dest_root, 'test')

    # Vérification dossier source
    if not os.path.exists(source_dir):
        print(f"ERREUR : Le dossier source Domain_B est introuvable ici :\n{source_dir}")
        return

    # 4. Création des dossiers cibles
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 5. Récupération des images (récursif pour inclure les sous-dossiers 1, 2, 10, etc.)
    all_images = []
    print(f"Recherche des images dans {source_dir}...")
    
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                all_images.append(os.path.join(root, file))

    total_images = len(all_images)
    if total_images == 0:
        print("Aucune image trouvée dans Domain_B.")
        return

    # 6. Mélange et split (80% Train / 20% Test)
    random.shuffle(all_images)
    train_ratio = 0.80
    split_idx = int(total_images * train_ratio)
    
    train_images = all_images[:split_idx]
    test_images = all_images[split_idx:]

    print(f"Total images : {total_images}")
    print(f" -> Train (80%) : {len(train_images)}")
    print(f" -> Test  (20%) : {len(test_images)}")

    # 7. Fonction de copie
    def copy_files(file_list, destination, label):
        print(f"Copie vers {label} en cours...")
        count = 0
        for f in file_list:
            try:
                # On ne garde que le nom du fichier, pas l'arborescence (ex: 1/img.jpg -> train/img.jpg)
                shutil.copy2(f, os.path.join(destination, os.path.basename(f)))
                count += 1
            except Exception as e:
                print(f"Erreur sur {f}: {e}")
        return count

    c_train = copy_files(train_images, train_dir, "TRAIN")
    c_test = copy_files(test_images, test_dir, "TEST")

    print(f"\nTerminé ! {c_train + c_test} fichiers copiés dans {dest_root}")

if __name__ == "__main__":
    split_domain_b()