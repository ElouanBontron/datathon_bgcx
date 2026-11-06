# README — Tests et résultats (branche tarik-test)

Ce fichier récapitule tous les tests effectués pour optimiser le notebook
`notebooks/timestral.ipynb` (Timestral-209M zero-shot + ensemble GBM).

## Protocole d'évaluation

- **Backtest** : `split(df, t0)` rejoue le challenge sur les vendredis de `ORIGINES`
  (2024-11 → 2026-10), prédiction du samedi au mercredi suivant (gap 1j + 120h).
- **Métrique** : RMSE par arc, puis moyenne. Focus principal sur **Convention +
  Saints-Pères** (Champs-Élysées souvent absent du test, capteur invalide).
- **Holdout temporel** : sélection des hyperparamètres sur 2024–2025, évaluation
  finale sur **2026 non vu** pour détecter le surapprentissage.

## Résultats clés (RMSE mean/date, Convention + Saints-Pères)

| Config | 2024–25 (sélection) | **2026 (holdout)** |
|--------|--------------------:|-------------------:|
| naif lag-7j | 92.5 | 79.1 |
| Timestral seul | 67.5 | 60.3 |
| LightGBM seul | 64.3 | 65.5 |
| XGBoost seul | — | 67.7 |
| **Timestral + 35% LGB** | **64.2** | **58.3** ← retenu |

## Ce qui a été testé et rejeté

| Test | Résultat |
|------|----------|
| Covariables supplémentaires (profil_normal, lag_an, météo, fermeture_prevue…) | ❌ pire (83.8 vs 83.4 sur 3 arcs) |
| Contexte 1024h / 4096h | ❌ pire que 2048h (85.4 / 84.8) |
| Encodage cyclique heure/jour (sin/cos) | ✅ aide Champs-Élysées, neutre sur les 2 autres — **retenu** |
| XGBoost comme backend de l'ensemble | ❌ LGB gagne à chaque poids (58.9 vs 59.5 à w=0.35) |
| Ensemble 3 voies Timestral+LGB+XGB | ❌ XGB redondant (w_xgb=0 optimal partout) |
| Fine-tuning de Timestral (head 4.9M, 140 fenêtres, 4 epochs) | ❌ overfit : 66.6 vs 65.2 zero-shot sur holdout 2026 |
| Poids par arc (Conv 0.5 / SP 0.3) | ⚠️ gain marginal (61.8 vs 62.0), risque d'overfit — poids uniforme retenu |

## Choix finaux (dans `notebooks/timestral.ipynb`)

- **Timestral-209M zero-shot**, CTX=2048, H=144 (gap 24h + horizon 120h)
- Covariables connues : calendrier (férié, pont, paris_respire, événement,
  confinement) + cyclique (heure_sin/cos, jour_sin/cos) + catégorielles
  (vacances_code, type_jour)
- **Ensemble : 65% Timestral + 35% LightGBM** (`W_GBM=0.35`, `BACKEND="lgb"`)
- Poids optimal robuste : plateau w=0.25–0.5, 0.35 au milieu (confirmé holdout 2026)

## Scripts de test (reproductibles)

| Fichier | Description |
|---------|-------------|
| `test_timestral_variants.py` | Backtest des variantes Timestral (covariables, CTX) |
| `test_ensemble.py` | Sweep des poids LGB dans l'ensemble |
| `test_focus_2arcs.py` | Focus Convention+Saints-Pères uniquement |
| `test_holdout.py` | Holdout temporel 2024-25 vs 2026 |
| `test_blend_lgb_xgb.py` | Comparaison LGB vs XGB comme backend |
| `test_threeway.py` | Grid search ensemble 3 voies (ts+lgb+xgb) |
| `test_xgb_fair.py` | XGB vs LGB sous le même protocole |
| `test_finetune_full.py` | Fine-tuning partiel de Timestral + évaluation holdout |

Lancer depuis la racine, ex. :
```bash
python test_holdout.py
```
