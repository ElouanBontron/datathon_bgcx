# Idées pour le projet

Pour l'instant on a pu obtenir un premier dataset pour pouvoir au moins tester différents modèles et approches.

Les pistes que j'imagine pour pouvoir avancer :

## I – Améliorer nos données d'entrainement

- Y a plusieurs points importants à remarquer, on va devoir prédire les données du **07 au 11 novembre**, en particulier y a : **un week-end et un jour férié**, donc notre modèle faudra qu'il soit robuste là-dessus : mettre de l'importance pour pas qu'il foire la prédiction du 11 novembre de l'année dernière par exemple.
- Vu que j'ai tout téléchargé dans mon PC, on pourra peut-être entrainer le modèle sur les **arcs proches** de ceux qui nous intéressent, on aura sans doute des valeurs similaires afin d'avoir plus de données ! _(à faire si on a du temps jpense)_
- Essayer de récupérer d'autres **informations open source** qui peuvent être utiles pour prédire le débit etc :
  - **Météo** (grv important, faudra qu'on trouve du coup un modèle open source qui prédit la météo pour pouvoir prédire les valeurs qu'il fera pour des jours futurs)
  - Infos sur les routes
  - Vacances et jours fériés
  - Covid
  - Prix de l'essence, utilisation des transports en commun
  - Événements majeurs

## II – Le modèle de prédiction

Une fois qu'on aura des données d'entrées propres on pourra faire notre entrainement et notre test pour choisir le modèle le plus pertinent. Je m'y connais pas trop dans cette phase…

Parmi les trucs qu'on pourra tester y a surtout :

- On entraine un modèle par arc ou un modèle qui prédit tous les arcs ?
- **Modèle de base qui sert de référence** : mettre la valeur précédente en prédiction, ça permet de voir le RMSE d'un modèle pourri pour comparer
- **LightGBM / XGBoost** : askip c'est souvent utilisé (d'après ChatGPT, à creuser)
- **Faire des mélanges de différents modèles**

Un truc cool qu'on pourrait faire c'est d'essayer quelques modèles différents et bien les adapter, et après faire un mix, et askip ça marche mieux qu'un seul modèle.

Et dernier truc : bien vérifier que notre pred du **débit > 0** et **taux d'occupation entre 0 et 100 %**.

## III – Partie Business

J'en ai aucune idée…
