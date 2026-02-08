# ActiveLearning_Solo

# CR

# Sepration des data en train et en test
## Domain A
* 25 % des data sont pour le test
* 75 % des data sont pour le train

## Domain B
* 20 % des data sont pour le test
* 80 % des data sont pour le train

## Train Domain A

courbe d'apprentissage :

## Test du domain A

Graphique + MSE Loss

## Test du domain B avec uniquement entrainement sur le domain A

Graphique + MSE Loss

## Comparaison

Expliquer que l'entrainement sur le domain A est efficace mais pas suffisant pour apprendre par rapport a un seul environnement, donc mettre en place un domain B pour améliorer l'apprentissage

# Mise en place d'un systeme d'active learning
## Random

* Ajout de 1% des data du domain B au domain A pour train
* Ajout de 1% des data pour arriver a 2% du domain B au domain A pour le train
* Ajout de 3% des data pour arriver a 5% du domain B au domain A pour le train
* Ajout de 5% des data pour arriver a 10% du domain B au domain A pour le train
* Ajout de 10% des data pour arriver a 20% du domain B au domain A pour le train
* Ajout de 30% des data pour arriver a 50% du domain B au domain A pour le train

Grapgique de courbe d'apprentissage

## Test du random

Benchmark A vs B et explication

Explication 