"""Données externes Paris 2014 -> 2026, à lancer depuis data/external/. Jointures :
- meteo_paris.csv       (1 ligne / heure)        sur datetime_utc
- calendrier_paris.csv  (1 ligne / jour)         sur date (jour en heure de Paris)
- evenements_paris.csv  (1 ligne / jour / axe)   sur date + axe

meteo_paris.csv (Open-Meteo ERA5) :
- datetime_utc         : heure UTC (même format que les datasets trafic)
- temperature_2m       : température de l'air à 2 m (°C)
- relative_humidity_2m : humidité relative à 2 m (%)
- precipitation        : pluie + neige tombées sur l'heure précédente (mm)
- snowfall             : neige tombée sur l'heure précédente (cm)
- cloud_cover          : couverture nuageuse (%, 0 = ciel clair, 100 = couvert)
- wind_speed_10m       : vitesse du vent à 10 m (km/h)
- weather_code         : code WMO du temps (0 ciel clair, 1-3 nuageux, 45-48 brouillard,
                         51-55 bruine, 61-65 pluie, 71-75 neige, 80-82 averses, 95+ orage)

calendrier_paris.csv :
- date          : jour (heure de Paris)
- ferie         : True si jour férié (calendrier.api.gouv.fr)
- pont          : True si vendredi après un jeudi férié ou lundi avant un mardi férié
- vacances      : True si vacances scolaires zone C (Paris)
- nom_vacances  : nom des vacances (Toussaint, Noël...), vide sinon

evenements_paris.csv :
- date, axe     : jour (heure de Paris) et axe
- paris_respire : True si Champs-Élysées piétons (1er dimanche du mois depuis mai 2016, ~10h-18h)
- evenement     : True si événement sur les Champs (8 mai, 14 juillet, 11 nov., 31 déc., arrivée du Tour,
                  finale Coupe du monde 2018, samedis gilets jaunes nov. 2018 - mars 2019)
- confinement   : True si confinement Covid (tous les axes)
- chantier      : nb de chantiers sur la chaussée à moins de 150 m de l'arc (Open Data Paris ;
                  trou possible pour les chantiers commencés ET finis entre 2024 et 2026)
"""
import pandas as pd

JOURS = pd.date_range("2014-01-01", "2026-12-31")


def meteo():
    vars_ = "temperature_2m,relative_humidity_2m,precipitation,snowfall,cloud_cover,wind_speed_10m,weather_code"
    url = ("https://archive-api.open-meteo.com/v1/archive?latitude=48.8566&longitude=2.3522"
           f"&start_date=2014-01-01&end_date=2026-10-31&timezone=GMT&format=csv&hourly={vars_}")
    df = pd.read_csv(url, skiprows=3).rename(columns=lambda c: c.split(" (")[0])
    df["datetime_utc"] = pd.to_datetime(df.pop("time"), utc=True)
    df.to_csv("meteo_paris.csv", index=False)


def calendrier():
    feries = pd.read_json("https://calendrier.api.gouv.fr/jours-feries/metropole.json", typ="series")
    df = pd.read_csv("https://raw.githubusercontent.com/AntoineAugusti/vacances-scolaires/master/data.csv", parse_dates=["date"])
    df = df[df.date.between("2014-01-01", "2026-12-31")].rename(columns={"vacances_zone_c": "vacances"})
    df["ferie"] = df.date.isin(feries.index)
    jour = df.date.dt.day_of_week  # 0 = lundi, 4 = vendredi
    df["pont"] = (jour.eq(4) & df.ferie.shift(1, fill_value=False)) | (jour.eq(0) & df.ferie.shift(-1, fill_value=False))
    df[["date", "ferie", "pont", "vacances", "nom_vacances"]].to_csv("calendrier_paris.csv", index=False)


def evenements():
    arcs = {"Champs-Elysees": (2.3017, 48.8715), "Convention": (2.2932, 48.8386), "Saints-Peres": (2.3325, 48.8573)}
    url = ("https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/{}/exports/csv?select=num_emprise,date_debut,date_fin"
           "&where=within_distance(geo_point_2d,geom'POINT({} {})',150m) and localisation_detail='EMPRISE_CHAUSSEE'")
    tour = ["2014-07-27", "2015-07-26", "2016-07-24", "2017-07-23", "2018-07-29", "2019-07-28", "2020-09-20",
            "2021-07-18", "2022-07-24", "2023-07-23", "2025-07-27"]
    confinements = [("2020-03-17", "2020-05-10"), ("2020-10-30", "2020-12-14"), ("2021-04-03", "2021-05-02")]
    j = JOURS
    gilets_jaunes = j[(j.day_of_week == 5) & (j >= "2018-11-17") & (j <= "2019-03-16")]
    events = j[j.strftime("%m-%d").isin(["05-08", "07-14", "11-11", "12-31"])].union(pd.to_datetime(tour + ["2018-07-15"])).union(gilets_jaunes)

    out = []
    for axe, (lon, lat) in arcs.items():
        champs = axe == "Champs-Elysees"
        c = pd.concat(pd.read_csv(url.format(ds, lon, lat).replace(" ", "%20"), sep=";", parse_dates=["date_debut", "date_fin"])
                      for ds in ["chantiers-a-paris-copie3", "chantiers-a-paris"]).drop_duplicates("num_emprise")
        out.append(pd.DataFrame({
            "date": j, "axe": axe,
            "paris_respire": champs & (j.day_of_week == 6) & (j.day <= 7) & (j >= "2016-05-08"),
            "evenement": champs & j.isin(events),
            "confinement": sum((j >= a) & (j <= b) for a, b in confinements).astype(bool),
            "chantier": [((c.date_debut <= d) & (c.date_fin >= d)).sum() for d in j],
        }))
    pd.concat(out).to_csv("evenements_paris.csv", index=False)


if __name__ == "__main__":
    meteo()
    calendrier()
    evenements()
