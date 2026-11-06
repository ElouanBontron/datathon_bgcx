"""XGBoost de Lucie (branche lucie/xgboost, modele_xgboost_par_axe.ipynb), repris tel quel :
un modèle par arc et par cible, axe retiré des features, vacances en one-hot, mêmes hyperparamètres.
Seule différence : il reçoit nos données (imputées) et nos FEATURES actuelles.
Lancer depuis la racine : uv run python model/modele_xgboost.py  -> test final + model/output_xgboost.csv"""
import pandas as pd
from commun import lancer
from dataset_create import FEATURES
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

model_features = [f for f in FEATURES if f != "axe"]
categorical_features = ["vacances"]
numeric_features = [f for f in model_features if f not in categorical_features]


def create_model():  # copie de son notebook
    preprocessor = ColumnTransformer(transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                                                   ("num", "passthrough", numeric_features)])
    regressor = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                             objective="reg:squarederror", random_state=42, n_jobs=-1)
    return Pipeline([("preprocessor", preprocessor), ("xgb", regressor)])


def predire(full, t0, cible):
    X = full[FEATURES].astype({c: float for c in FEATURES if full[c].dtype == bool})
    out = []
    for axe in X.axe.unique():
        train = (full.datetime < t0) & full[cible].notna() & (X.axe == axe)
        futur = (full.datetime >= t0) & (X.axe == axe)
        m = create_model().fit(X.loc[train, model_features], full.loc[train, cible])
        out.append(pd.Series(m.predict(X.loc[futur, model_features]), X.index[futur]))
    return pd.concat(out)


if __name__ == "__main__":
    lancer(predire, "xgboost")
