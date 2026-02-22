import subprocess
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent

scripts_to_run = [
    "active_learning_mixed_sequential.py",
    "active_learning_mixed_integrated.py"
]

print("="*60)
print("🚀 DÉMARRAGE DU PIPELINE DES STRATÉGIES MIXTES")
print("="*60)

for script_name in scripts_to_run:
    script_path = script_dir / script_name
    
    if not script_path.exists():
        print(f"❌ AVERTISSEMENT : Le script '{script_name}' est introuvable.")
        continue
        
    print("\n" + "="*50)
    print(f"▶️  EN COURS : {script_name}")
    print("="*50 + "\n")
    
    try:
        subprocess.run([sys.executable, str(script_path)], check=True)
        print(f"\n✅ SUCCÈS : '{script_name}' s'est terminé correctement.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERREUR : '{script_name}' a planté (Code : {e.returncode}).")
        break 
    except KeyboardInterrupt:
        print("\n🛑 INTERRUPTION MANUELLE.")
        break

print("\n🎉 ENTRAÎNEMENTS HYBRIDES TERMINÉS !")