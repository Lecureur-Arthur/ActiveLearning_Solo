import subprocess
import sys
from pathlib import Path

# ==========================================
# 1. CONFIGURATION
# ==========================================
# On récupère le dossier où se trouve CE script
script_dir = Path(__file__).resolve().parent

# Liste de vos scripts exacts, dans l'ordre où vous voulez les exécuter
scripts_to_run = [
    "active_learning_random_robust.py",
    "active_learning_mc_dropout.py",
    "active_learning_learning_loss.py",
    "active_learning_kmeans.py",
    "active_learning_outliers.py"
    # Vous pouvez aussi ajouter votre script "benchmark_all_strategies.py" à la fin si vous l'avez !
]

print("="*60)
print("🚀 DÉMARRAGE DU PIPELINE COMPLET D'ACTIVE LEARNING")
print("="*60)
print(f"Dossier de travail : {script_dir}")
print(f"Nombre de stratégies à lancer : {len(scripts_to_run)}\n")

# ==========================================
# 2. EXÉCUTION SÉQUENTIELLE
# ==========================================
for script_name in scripts_to_run:
    script_path = script_dir / script_name
    
    # Vérification que le fichier existe bien avant de le lancer
    if not script_path.exists():
        print(f"❌ AVERTISSEMENT : Le script '{script_name}' est introuvable. On passe au suivant.")
        continue
        
    print("\n" + "="*50)
    print(f"▶️  EN COURS : {script_name}")
    print("="*50 + "\n")
    
    try:
        # sys.executable garantit qu'on utilise le même environnement Python (ex: votre venv ou conda)
        # check=True fait planter proprement le script si le sous-script plante
        subprocess.run([sys.executable, str(script_path)], check=True)
        
        print(f"\n✅ SUCCÈS : '{script_name}' s'est terminé correctement.")
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERREUR CRITIQUE : '{script_name}' a planté (Code d'erreur : {e.returncode}).")
        print("Arrêt du pipeline pour vous permettre de vérifier l'erreur.")
        break # On arrête la boucle pour ne pas lancer la suite sur des données corrompues
        
    except KeyboardInterrupt:
        print("\n🛑 INTERRUPTION MANUELLE : Vous avez arrêté le script (Ctrl+C).")
        break

print("\n" + "="*60)
print("🎉 PIPELINE TERMINÉ !")
print("="*60)