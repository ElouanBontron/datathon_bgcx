# Dataset trafic : 3 arcs, 2014 → 2026

**Un fichier par axe** : `champs_elysees.csv`, `convention.csv`, `saints_peres.csv` (112 488 lignes chacun, 337 464 au total)

Mesures horaires des capteurs de la Ville de Paris sur les 3 arcs du challenge,
du **01/01/2014 au 01/11/2026**. Une ligne = un axe × une heure. La grille horaire est complète :
s'il n'y a pas de mesure, la ligne existe quand même avec `debit` / `occupation` vides.

| Axe | Arc (`arc_id`) | Nœud amont → aval |
|---|---|---|
| Champs-Elysees | 4264 | Av_Champs_Elysees-Washington → Av_Champs_Elysees-Berri |
| Convention | 5671 | Lecourbe-Convention → Convention-Blomet |
| Saints-Peres | 191 | Sts_Peres-Voltaire → Sts_Peres-Universite |

## Colonnes

| Colonne | Description |
|---|---|
| `axe`, `arc_id` | voir tableau ci-dessus |
| `datetime_utc` | heure de la mesure en UTC, **seule colonne de date** (l'heure de Paris se recalcule à l'import, voir ci-dessous). Désigne la **fin** de l'heure mesurée (« 09:00 » = trafic de 8h à 9h) |
| `debit` | **cible 1**, nombre de véhicules sur l'heure |
| `occupation` | **cible 2**, % du temps où un véhicule est sur le capteur (<15 % fluide, 15-30 % pré-saturé, 30-50 % saturé, ≥50 % bloqué) |
| `etat_trafic` | Fluide / Pré-saturé / Saturé / Bloqué / Inconnu |
| `etat_arc` | Ouvert / Barré / Invalide (données récentes uniquement) |
| `etat_arc_code` | code brut avant oct. 2025 : 2 = Barré, 3 = Ouvert *ou* Invalide (peu fiable) |
| `ligne_absente` | True si l'heure n'existait pas du tout dans la source |
| `source` | `old_data` (archives open data, jusqu'au 30/09/2025) ou `recent` (export du challenge, depuis le 01/10/2025) |

## Importer

```python
from dataset_create import charger

df = charger()                    # les 3 axes ; charger("Convention") pour un seul
# Équivalent à la main :
#   df = pd.read_csv("data/processed_data/convention.csv", low_memory=False)
#   df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
# Heure de Paris : à utiliser pour les features (heure, jour de semaine...), le trafic suit l'heure locale.
# Garder datetime_utc pour les jointures et les décalages (lags) : unique même au changement d'heure.
# df["datetime"] = df["datetime_utc"].dt.tz_convert("Europe/Paris")  (déjà fait par charger)

# Exemple : une série par axe et par cible
debit = df.pivot(index="datetime", columns="axe", values="debit")
```

## À savoir avant de modéliser

- **Champs-Élysées : capteur en panne depuis le 19/11/2025.** Aucune mesure depuis presque un an.
  Il faudra un profil historique ou un arc voisin pour prédire cet axe.
- **Beaucoup de trous récents** : débit manquant 58 % (2023) et 72 % (2024) sur Convention ;
  46 % (2024) et 80 % (2025) sur Saints-Pères. Avant 2022, moins de 3 % de trous en général.
- **Le niveau de trafic change dans le temps** : Champs-Élysées passe de ~1 470 véh/h en moyenne (2014)
  à ~740 (2025). Préférer les années récentes pour les niveaux ; les anciennes servent pour la forme
  des profils (ex. effet des jours fériés).
- **2020-2021 = Covid** (confinements) : à exclure ou à signaler par une variable.
- **À prédire : du 07/11 au 11/11/2026**, le 11/11 est férié. Format attendu par BCG (360 lignes) :
  colonnes `arc` (`Champs-Elysées`, `Convention`, `Saint-Pères`), `datetime` (`"%Y-%m-%d %H:%M"`),
  `debit_horaire`, `taux_occupation`.

## Comment il a été construit

`dataset_create.py` à la racine (`uv run python dataset_create.py --jobs 4`, ~5 min) :

- `dataset_annee(annee)` : lit `data/unprocessed_data/<annee>/` (archives `*.txt` 2014-2025 ou
  exports du challenge `*.csv` dans `2026/`), garde les 3 arcs, renvoie la grille horaire complète.
- `dataset_complet()` : appelle `dataset_annee` pour chaque année et fusionne.
  Sur la période commune (oct.-déc. 2025) les deux sources sont identiques (écart 0) ; on garde la source récente.
- `sauvegarder_par_axe(df)` : écrit les 3 CSV.

Les archives 2014-2025 (~75 Go) ne sont pas versionnées : à placer dans `data/unprocessed_data/<année>/`.
Définitions des champs : notice Ville de Paris *Données de trafic issues des capteurs permanents* (juin 2019).
