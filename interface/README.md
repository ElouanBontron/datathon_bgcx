# FlowPilot — interface de livraison

FlowPilot transforme les prévisions horaires du challenge en créneaux de livraison, planning et simulation de gains. Une page regroupe trois onglets : **Vue d’ensemble**, **Impact** et **Livraisons**.

## Télécharger le projet

Avec Git :

```sh
git clone https://github.com/ElouanBontron/datathon_bgcx.git
cd datathon_bgcx
```

Sans Git, ouvrir [le dépôt GitHub](https://github.com/ElouanBontron/datathon_bgcx), choisir **Code → Download ZIP**, puis extraire l’archive. Ouvrir le dossier contenant `app.py`, `requirements.txt` et `data/`.

Les CSV préparés et le fond de carte sont fournis dans le dépôt. Aucun téléchargement des archives brutes historiques n’est nécessaire pour utiliser l’interface. La documentation scientifique et celle des données restent disponibles dans le [README du dépôt](../README.md) et le [README des données préparées](../data/processed_data/README.md).

## Lancer l’application

Python 3.11 ou plus récent est nécessaire. Sous Windows, double-cliquer sur **`Lancer_FlowPilot.bat`**, puis ouvrir **http://localhost:8501**. Garder la fenêtre de commande ouverte pendant l’utilisation ; `Ctrl+C` arrête le serveur.

Installation manuelle sous Windows, depuis PowerShell à la racine du dépôt :

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py --server.address localhost
```

Sous Linux ou macOS :

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py --server.address localhost
```

La première installation nécessite Internet. L’environnement `.venv` reste local. Pour l’interface, installer `requirements.txt` suffit : les outils de modélisation du `pyproject.toml` scientifique ne sont pas tous nécessaires.

## Utiliser les trois onglets

Choisir la journée et les axes en haut de la page ; cette sélection est commune aux trois onglets.

- **Vue d’ensemble** : voir les économies quotidiennes estimées, le temps gagné, la variation de coût et le départ recommandé, puis explorer la carte de Paris et les meilleurs créneaux.
- **Impact** : comparer le départ habituel au départ optimisé et ajuster volumes, durées et coûts. Ces paramètres actualisent aussi les gains de la vue d’ensemble. Exporter la simulation en CSV.
- **Livraisons** : saisir les demandes et leurs fenêtres de disponibilité, ajuster la flotte et construire le planning. Consulter sa durée et son coût estimés, puis télécharger le CSV ou le fichier agenda `.ics`.

Les demandes impossibles à planifier restent visibles avec une explication. Les horaires sont ceux de Paris. L’export agenda s’importe dans Outlook, Google Agenda ou Apple Calendrier.

## Importer les prévisions de l’équipe

Ouvrir **Réglages** en haut de la page, choisir **CSV de l’équipe** dans **Moteur utilisé**, puis importer un fichier UTF-8 au format BCG :

```csv
arc,datetime,debit_horaire,taux_occupation
Champs-Elysées,2026-11-07 00:00,320,7.5
Convention,2026-11-07 00:00,140,4.2
Saint-Pères,2026-11-07 00:00,85,3.1
```

Ces chiffres illustrent uniquement le format. Chaque journée utilisée doit contenir **72 lignes : 3 axes × 24 heures**, de `00:00` à `23:00`, sans doublon. L’horizon complet du 7 au 11 novembre 2026 représente 360 lignes. Un fichier peut couvrir plusieurs jours.

`datetime` désigne le **début du créneau en heure de Paris** : `2026-11-07 00:00` correspond à 00 h–01 h. Les dates ISO avec décalage explicite sont acceptées si ce décalage correspond à Paris. Le backend décale les mesures historiques sources d’une heure, car elles désignent la fin de l’heure mesurée ; il ne décale pas les prédictions importées.

Les noms avec ou sans accents, les séparateurs virgule ou point-virgule et le BOM UTF-8 sont acceptés. Le débit doit être positif ou nul et l’occupation comprise entre 0 et 100. L’import valide les dates, les valeurs et les doublons ; il n’invente aucune précision du modèle.

## Interpréter les résultats

Les données historiques sont réelles : **337 464 lignes horaires** couvrent les trois segments étudiés de 2014 à 2026, avec les heures manquantes conservées. Les segments sont Champs-Élysées (arc 4264), Convention (5671) et Saints-Pères (191), pas l’ensemble des rues parisiennes. Leurs géométries proviennent du challenge. Les contours des arrondissements sont stockés localement ; la Seine est un repère schématique.

Le dépôt contient des essais de modèles, sans modèle final entraîné prêt à déployer dans la version initialement intégrée. Par défaut, l’application utilise un **profil historique saisonnier** : médianes d’heures comparables antérieures au jour choisi, avec prise en compte des jours fériés, dont le 11 novembre. Ce profil est une référence statistique, sans performance prédictive ni intervalle de confiance calibré affiché. Le CSV permet de le remplacer par les prévisions finales.

Le taux d’occupation mesure la part du temps où un véhicule occupe le capteur ; le débit mesure les véhicules par heure. La densité physique en véhicules/km n’est pas mesurée. Le capteur des Champs-Élysées ne fournit plus de mesures récentes depuis novembre 2025 : son profil repose sur des observations plus anciennes.

Les créneaux minimisent le score `0,85 × occupation + 15 × débit / maximum quotidien de l’axe`, moyenné sur les axes et les heures du créneau. Ce score n’est pas une probabilité de réussite.

Les gains sont des **simulations**, selon `trajet = durée fluide × (1 + occupation / 50)`. Le coût comprend le temps de trajet et de service au coût équipe, plus le coût véhicule pendant la conduite. La projection annuelle répète le scénario sur le nombre de jours comparables choisi. Cette relation n’est pas calibrée sur des temps de parcours mesurés : les montants ne sont ni des gains déjà réalisés ni une garantie du modèle.

Le planning respecte les fenêtres et la capacité des véhicules. Il estime chaque trajet indépendamment, sans calcul de tournée entre adresses ni prise en compte des restrictions de voirie ou du stationnement.

## Mettre à jour le dépôt

Avec une copie Git propre, exécuter :

```sh
git pull --ff-only
```

Sous Windows, `actualiser_donnees.ps1` effectue aussi une mise à jour contrôlée :

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\actualiser_donnees.ps1
```

Dans un clone standard, le script actualise le dépôt racine et peut donc actualiser `app.py`. Il refuse les modifications locales non enregistrées et les mises à jour nécessitant une fusion. Dans l’organisation de travail initiale, où l’interface se trouve à côté d’une copie scientifique dans `upstream/`, il actualise uniquement cette copie scientifique. Il n’effectue aucune suppression ni réinitialisation. Redémarrer l’application après une mise à jour.

Avec Download ZIP, télécharger une nouvelle archive et l’extraire dans un autre dossier pour conserver les éventuelles modifications locales.

## Notebook et vérifications


Depuis la racine, lancer les tests des données, du calendrier, de l’import CSV, du planning et des calculs :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Sous Linux/macOS, remplacer `.\.venv\Scripts\python.exe` par `.venv/bin/python`. La suite comprend au moins les 24 vérifications initiales, complétées au fil des évolutions.

Pour les parcours navigateur, installer les dépendances de développement et lancer l’application avant les tests ; Chrome doit être installé :

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe tests/browser_smoke.py
.\.venv\Scripts\python.exe tests/browser_scenarios.py
```

Les tests vérifient les trois onglets, le partage des gains, les filtres, l’import CSV et l’export agenda. Leurs données synthétiques ne sont pas conservées dans l’application. Les aperçus publiés se trouvent dans [`artifacts/flowpilot_overview.png`](artifacts/flowpilot_overview.png), [`artifacts/flowpilot_impact.png`](artifacts/flowpilot_impact.png) et [`artifacts/flowpilot_planning.png`](artifacts/flowpilot_planning.png).
