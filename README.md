# Eye Tracking via Active Learning : Adaptation de Domaine et Régression sur Images

Ce projet explore l'efficacité des stratégies d'**Apprentissage Actif (Active Learning)** pour résoudre le problème du **décalage de distribution (Domain Shift)** dans un contexte de régression visuelle (prédiction des coordonnées du regard sur un écran à partir d'une simple webcam).

## 1. Contexte et Objectifs

L'objectif principal est d'adapter un modèle de régression profond (ResNet-18), initialement entraîné dans un environnement très contrôlé (**Domaine A**), vers de nouveaux environnements variables (**Domaine B** : nouveaux visages, lumières différentes, port de lunettes).

Pour éviter d'avoir à ré-annoter manuellement des milliers d'images pour chaque nouvel utilisateur, nous utilisons des algorithmes d'**Active Learning**. L'idée est de laisser l'IA sélectionner intelligemment une infime fraction (1% à 5%) des images les plus "utiles" du Domaine B pour se ré-entraîner (Fine-Tuning), surpassant ainsi une sélection purement aléatoire.

### Qu'est-ce que le Fine-Tuning ?

Le **Fine-Tuning** (ou ajustement fin) est une technique consistant à prendre un modèle d'IA pré-entraîné sur une vaste base de données (ici ImageNet) et à le spécialiser pour une tâche précise. Dans ce projet, nous n'apprenons pas au modèle "comment voir" depuis zéro, mais nous adaptons ses connaissances pour qu'il devienne expert sur notre tâche d'Eye Tracking.

---

## 2. Protocole Expérimental et Données

### 2.1. Architecture du Modèle

* **Modèle** : ResNet-18 pré-entraîné.
* **Tâche** : Régression spatiale des coordonnées (X, Y) du regard.
* **Fonction de perte (Loss)** : MSE (Mean Squared Error).
* **Résolution d'entrée** : 1920x1080 (Full HD).

### 2.2. Jeux de Données (Datasets)

Les données sont réparties en deux domaines distincts pour simuler le changement d'environnement :

| Domaine | Rôle | Train Split | Test Split | Description |
| --- | --- | --- | --- | --- |
| **Domaine A (Source)** | Apprentissage Initial | **75%** | **25%** | Environnement de référence parfait (1 individu, bonne lumière, fond neutre). |
| **Domaine B (Cible)** | Adaptation & Active Learning | **80%** | **20%** | Nouveaux environnements complexes. Le set "Train" sert de réservoir (*Pool*) non labellisé pour l'Active Learning. |

---

## 3. Phase 1 : Entraînement Initial & Domain Gap

### 3.1. Entraînement Supervisé (Domaine A)

Le modèle est d'abord entraîné exclusivement sur le Domaine A. La courbe d'apprentissage montre une convergence rapide de l'erreur d'entraînement et de validation, validant notre choix d'architecture (ResNet-18). L'entraînement est arrêté à 20 époques via un *Early-Stopping* pour éviter le sur-apprentissage.

### 3.2. Mise en évidence du "Domain Gap"

En testant ce modèle directement sur les données inédites du Domaine B (sans ré-entraînement), nous observons une chute drastique des performances :

* **MSE sur Domaine A (Test)** : `~0.0015` (Précision excellente)
* **MSE sur Domaine B (Test)** : `~0.0891` (Erreur élevée)

Cette augmentation significative de l'erreur prouve l'existence d'un **Domain Shift**. Le modèle ne généralise pas face aux nouvelles conditions d'éclairage ou aux nouveaux visages. Une adaptation est indispensable.

---

## 4. Phase 2 : Stratégies d'Active Learning

Pour adapter le modèle au Domaine B de manière économique, nous simulons l'ajout progressif de données annotées selon différents budgets (1%, 2%, 5%, 10%, 20%, 50%). Nous comparons plusieurs stratégies intelligentes à une sélection aléatoire de référence (*Random Robust*).

### 4.1. Baseline : Random Robust

* **Fonctionnement** : Sélectionne les images de manière totalement aléatoire. Pour lisser la variance statistique due au hasard, l'expérience est répétée plusieurs fois (N rounds) et les résultats sont moyennés pour obtenir une courbe de référence fiable avec son écart-type.

### 4.2. Famille "Incertitude" (Uncertainty Sampling)

Ces méthodes ciblent les images sur lesquelles le modèle "hésite" le plus.

* **MC Dropout** : Maintient la désactivation aléatoire des neurones (*Dropout*) active pendant l'inférence. Le modèle effectue plusieurs prédictions pour la même image. Plus les prédictions varient (forte variance), plus l'image est jugée incertaine et est sélectionnée pour être annotée.
* **Learning Loss** : Approche très élégante (Yoo et al., CVPR 2019) qui ajoute un module secondaire au réseau. Ce module a pour unique but de prédire la magnitude de l'erreur (la *Loss*) que le réseau principal va commettre sur une image. Les images avec la plus forte erreur prédite sont sélectionnées.

### 4.3. Famille "Diversité" (Diversity Sampling)

Ces méthodes forcent l'exploration spatiale du nouveau domaine pour éviter la redondance (ex: sélectionner 50 frames vidéo quasi identiques).

* **K-Means (Clustering)** : Extrait les caractéristiques profondes (*features*) de toutes les images non annotées et les regroupe en $K$ clusters (où $K$ est le budget d'images désiré). L'algorithme sélectionne ensuite l'image la plus proche du centre de chaque cluster (le centroïde), garantissant un panel d'images très variées.
* **Outliers (Isolement)** : Calcule la distance mathématique de chaque image non annotée par rapport à toutes les images déjà connues par le modèle. L'algorithme sélectionne les images les plus lointaines/isolées pour forcer le modèle à découvrir des situations inédites.

### 4.4. Famille "Stratégies Mixtes" (Hybrides)

Face au constat que la diversité seule surpasse souvent l'incertitude pour éviter le sur-apprentissage, nous combinons les deux approches :

* **Sequential Combination** : Agit en entonnoir. Présélectionne d'abord un grand groupe d'images difficiles grâce au *MC Dropout*, puis applique un *K-Means* uniquement sur ce sous-groupe pour en extraire les images à la fois difficiles ET différentes.
* **Integrated Scores** : Calcule une note sur 20 pour chaque image combinant 50% de son score d'incertitude (Dropout) et 50% de son score de distance spatiale (Outliers). Les images ayant la meilleure note globale sont sélectionnées.

---

## 5. Synthèse et Benchmark Final

L'évaluation de toutes ces stratégies révèle l'efficacité redoutable de l'Active Learning. Les stratégies basées sur la **diversité spatiale (K-Means)** et les **stratégies mixtes (Integrated Scores)** atteignent des performances optimales (MSE ~0.061) avec seulement **1% à 5% de données annotées**, surpassant de loin un modèle ré-entraîné sur 50% de données tirées au hasard.

Au-delà de 10% de budget, l'accumulation d'images temporelles redondantes provoque un phénomène de sur-apprentissage spécifique au domaine, d'où la remontée de l'erreur en forme de "U" sur le graphique global.

---

## 6. Architecture du Code et Exécution

Voici la marche à suivre pour reproduire les résultats du projet, de la préparation des données jusqu'à la génération des graphiques comparatifs :

1. **Split des données** :
Lancer `Python/IA/SplitTrainTest/DomainA/SplitDomainA.py` (puis le script correspondant pour le Domain B) afin de générer les dossiers `train` et `test`.
2. **Entraînement de la Baseline (Domaine A)** :
Lancer `Python/IA/Domain_A/train.py`.
3. **Mise en évidence du Domain Gap** :
Lancer `Python/IA/Domain_A/test_domainB.py`.
4. **Exécution de l'Active Learning** :
Utiliser le pipeline global pour exécuter les stratégies d'incertitude et de diversité :
`python Python/IA/Domain_B/run_pipeline.py`
5. **Exécution des Stratégies Mixtes** :
`python Python/IA/Domain_B/run_mixed_pipeline.py`
6. **Évaluation et Génération des Graphiques Finaux** :
Pour générer les graphiques lisibles et comparés à la Baseline Random :
`python Python/IA/Domain_B/evaluate_final_3_curves.py`