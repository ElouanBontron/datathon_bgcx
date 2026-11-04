# BCG X Datathon

## Données

Dans le dossier `data` on peut trouver 3 fichiers différents : `external`, `processed_data` et `unprocessed_data` :

- **`external`** correspond aux données externes ajoutées : météo, calendrier scolaire, évènements historiques majeurs
- **`unprocessed_data`** contient toutes les données du trafic routier de 2014 à 2026. Ces données ne sont pas disponibles sur le git car trop lourdes, mais peuvent être téléchargées [ici](https://opendata.paris.fr/explore/dataset/comptages-routiers-permanents-historique/information/)
- **`processed_data`** contient toutes les données du trafic routier pour les 3 arcs qui nous intéressent

Pour charger facilement les données, vous pouvez vous inspirer du fichier `data_explo.ipynb` en faisant par exemple :

```python
from dataset_create import charger
df = charger()
df.head()
```
