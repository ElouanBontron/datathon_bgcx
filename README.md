# FlowPilot : prévision du trafic parisien (BCG X Datathon 2026)

Ce projet a été réalisé dans le cadre du Datathon BCG X. L'objectif était de prévoir le **débit** ainsi que le
**taux d'occupation** de 3 rues de Paris (Champs-Élysées, Convention, Saints-Pères), du 07/11 au 11/11/2026.

Le débit et le taux d'occupation sont deux informations centrales de l'analyse routière : ils permettent de connaître
l'état actuel du trafic afin d'optimiser les déplacements en dehors des heures de pointe.

## Pipeline

```
data/unprocessed_data/   capteurs de la Ville de Paris (2014 → 2026)
        │  dataset_create.py              garde les 3 arcs, une ligne par arc et par heure
        ▼
data/processed_data/     un CSV par arc
        │  charger(imputer=True)          + calendrier et événements (data/external/)
        │                                 + trous du débit comblés grâce aux rues voisines (imputation_voisins.py)
        │  feature_engineering(df, t0)    18 features connues le jour J
        ▼
model/                   LightGBM + XGBoost + Timestral, moyenne des trois
        ▼
data/output/output_flowerrangers.csv
```

**Features** (`dataset_create.py`) : heure, jour de semaine, mois, type de jour (semaine / samedi / dimanche-férié),
vacances, férié, pont, événements (14 juillet, Paris Respire, manifestations…), confinement, même heure 7 et 14 jours
avant et leur moyenne sur 4 semaines, même jour un an avant, profil des 6 dernières semaines normales, et probabilité
de fermeture de la rue lors des éditions passées d'un événement.

**Modèles** (`model/`) :

| Fichier               | Modèle                                                                                                                                                                 |
| --------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `modele_lightgbm.py`  | LightGBM, un seul modèle pour les 3 arcs                                                                                                                               |
| `modele_xgboost.py`   | XGBoost, un modèle par arc                                                                                                                                             |
| `modele_timestral.py` | [Timestral-209M](https://github.com/mistralai/timestral) de Mistral, modèle pré-entraîné utilisé sans entraînement : il lit les 2048 dernières heures et le calendrier |
| `modele_ensemble.py`  | **modèle final** : moyenne des trois                                                                                                                                   |

Les arbres captent bien les jours atypiques (férié, manifestation) ; Timestral suit mieux la tendance récente. Leur moyenne
est plus fiable que chacun d'eux.

## Lancer

```bash
uv sync                                     # installe l'environnement
uv run python model/modele_ensemble.py      # test final + data/output/output_flowerrangers.csv (~10 min sur CPU)
uv run python tests/evaluer.py debit        # compare les modèles (ajouter --backtest pour le backtest, ~1 h)
```

Les CSV de `data/processed_data/` et `data/external/` sont fournis. Pour les reconstruire :
`uv run python dataset_create.py --jobs 4` (archives 2014-2025 à télécharger
[ici](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/information/), ~75 Go)
et `cd data/external && uv run python external_create.py`.

## Résultats

RMSE moyenne des arcs. Naïf = même heure 7 jours avant.

- **Semaine 2025** : on se place au vendredi 07/11/2025 et on prédit du samedi au mercredi 12/11 (11/11 férié inclus).
- **Test récent** : vraies valeurs du 01 au 06/11/2026 (Champs-Élysées absent : capteur en panne).
- **Backtest** : 23 semaines de nov. 2024 à oct. 2026, même découpage que le jour J.

| Débit (véh/h) | Naïf  | LightGBM | XGBoost | Timestral | **Ensemble** |
| ------------- | ----- | -------- | ------- | --------- | ------------ |
| Semaine 2025  | 143.1 | 64.4     | 67.4    | 105.2     | **69.3**     |
| Test récent   | 68.4  | 64.8     | 70.8    | 64.3      | **57.8**     |
| Backtest      | 110.9 | 87.7     | 87.2    | 83.9      | **81.7**     |

| Occupation (%) | Naïf | LightGBM | XGBoost | Timestral | **Ensemble** |
| -------------- | ---- | -------- | ------- | --------- | ------------ |
| Semaine 2025   | 3.66 | 2.99     | 2.96    | 3.23      | **2.88**     |
| Test récent    | 2.08 | 2.11     | 2.13    | 1.93      | **1.81**     |
| Backtest       | 3.70 | 2.99     | 3.02    | 2.60      | **2.74**     |

Sur le backtest, l'ensemble réduit l'erreur du naïf d'environ 26 % sur chacun des 3 arcs.

**Testé puis écarté** : météo (pluie, neige, gel : aucun gain), Chronos-2, Prophet, poids optimisés sur le backtest
(meilleurs sur les semaines normales, nettement moins bons sur la semaine avec férié).

## Organisation

```
dataset_create.py        construction, chargement et features
imputation_voisins.py    imputation du débit par les capteurs des rues voisines
model/                   modèles, évaluation commune (commun.py) et format du rendu (rendu.py)
tests/evaluer.py         comparaison des modèles sur les 3 tests
notebooks/               exploration des données et étude de l'imputation
data/                    données (voir data/processed_data/README.md)
```

## Interface client FlowPilot

Une page avec trois onglets : **Vue d'ensemble**, **Impact**, **Livraisons**. Carte de Paris, gains estimes, import des previsions CSV et planning exportable dans un agenda.

**Windows :** double-cliquer sur `interface/Lancer_FlowPilot.bat`, puis ouvrir http://localhost:8501.

**Installation manuelle (Python 3.11+) :**

```bash
python -m venv .venv
# Linux/macOS : source .venv/bin/activate
# Windows : .venv\Scripts\activate
cd interface
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

[Guide FlowPilot](interface/README.md) : installation, hypotheses, import CSV et tests. Le profil historique et les gains sont des estimations, en attendant les previsions finales de l'equipe.

![Vue d'ensemble de FlowPilot](interface/artifacts/flowpilot_overview.png)
