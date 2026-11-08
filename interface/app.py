"""FlowPilot : une page, trois onglets, un scénario commun de gains."""
from __future__ import annotations

from datetime import date
from html import escape
from pathlib import Path
import hashlib

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from flowpilot.charts import LABELS, paris_map, schedule_chart
from flowpilot.data import AXES, get_forecast
from flowpilot.imports import parse_predictions
from flowpilot.planning import build_schedule, estimate_impact, export_calendar, rank_slots

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="FlowPilot · Paris", page_icon="◈", layout="wide")
st.markdown(f"<style>{(ROOT / 'assets/styles.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)
forecast_for = st.cache_data(show_spinner=False)(get_forecast)
imported_data = st.cache_data(show_spinner=False)(parse_predictions)


def fmt(value, digits=0):
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


def french_day(day):
    days = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
    return f"{days[day.weekday()]} {day.day:02d} novembre {day.year}"


def metric_card(label, value, unit="", note="", featured=False):
    css = "metric featured" if featured else "metric"
    st.markdown(f'<div class="{css}"><div class="metric-label">{escape(label)}</div>'
        f'<div class="metric-value">{escape(str(value))} <small>{escape(unit)}</small></div>'
        f'<div class="metric-note">{escape(note)}</div></div>', unsafe_allow_html=True)


def section(title, badge=""):
    tag=f'<span class="pill">{escape(badge)}</span>' if badge else ""
    st.markdown(f'<div class="section-line"><h2>{escape(title)}</h2>'
        f'{tag}</div>', unsafe_allow_html=True)


def intro(title, subtitle):
    st.markdown(f'<div class="page-intro"><h1>{escape(title)}</h1>'
        f'<p class="hero-subtitle">{escape(subtitle)}</p></div>',unsafe_allow_html=True)


def plot(fig, key):
    st.plotly_chart(fig, width="stretch", key=key,
        config={"displayModeBar": False, "scrollZoom": False, "locale": "fr"})


def init_scenario(best_hour):
    defaults = {"scenario_deliveries":40, "scenario_baseline":9, "scenario_optimized":int(best_hour),
        "scenario_travel":20, "scenario_service":8, "scenario_cost":32.,
        "scenario_vehicle":4., "scenario_days":220}
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    return {"deliveries":int(st.session_state.scenario_deliveries),
        "baseline_hour":int(st.session_state.scenario_baseline),
        "optimized_hour":int(st.session_state.scenario_optimized),
        "free_flow_minutes":st.session_state.scenario_travel,
        "service_minutes":st.session_state.scenario_service,
        "hourly_cost":st.session_state.scenario_cost,
        "fuel_cost_per_hour":st.session_state.scenario_vehicle,
        "working_days":int(st.session_state.scenario_days)}


def gain_metrics(result, slots=None):
    columns = st.columns(4 if slots is not None else 3)
    with columns[0]: metric_card("Économies estimées / jour",fmt(result["daily_savings"]),"€", "Selon votre scénario",featured=True)
    with columns[1]: metric_card("Temps gagné / jour",fmt(result["time_saved_hours"],1),"h", "Sur l’ensemble des livraisons")
    with columns[2]: metric_card("Coût en moins",fmt(result["savings_pct"],1),"%", "Par rapport au départ habituel")
    if slots is not None:
        best=slots.iloc[0]
        with columns[3]: metric_card("Départ recommandé",f"{int(best.start_hour):02d}h–{int(best.end_hour):02d}h",note="Créneau le plus favorable · 07h–20h")


def slot_cards(slots):
    for index, row in enumerate(slots.head(3).itertuples(),start=1):
        st.markdown(f'<div class="slot-card"><span class="slot-rank">{index:02d}</span>'
            f'<div><div class="slot-time">{escape(row.label)}</div>'
            f'<div class="slot-desc">{escape(row.risk)} · {fmt(row.occupation,1)} % d’occupation</div></div></div>',unsafe_allow_html=True)


def show_overview(forecast, axes, slots, scenario, result):
    intro("Vos livraisons, au meilleur moment.",
        f"{scenario['deliveries']} livraisons · Départ habituel {scenario['baseline_hour']:02d}h → départ optimisé {scenario['optimized_hour']:02d}h")
    gain_metrics(result,slots)
    main, aside = st.columns([2.2,1],gap="large")
    with main:
        section("Paris, heure par heure")
        with st.container(border=True):
            hour=st.slider("Heure sur la carte",0,23,10,format="%02dh",key="map_hour")
            rows=forecast[forecast.axe.isin(axes) & forecast.datetime.dt.hour.eq(hour)]
            st.markdown(f'<div class="map-summary"><span>{hour:02d}:00 — {hour+1:02d}:00</span>'
                '<span><i style="background:#5a9871"></i>Fluide</span><span><i style="background:#d2a254"></i>Dense</span>'
                '<span><i style="background:#d47857"></i>Saturé</span><span><i style="background:#b65360"></i>Bloqué</span></div>',unsafe_allow_html=True)
            figure=paris_map(rows,AXES).update_layout(height=325)
            plot(figure,"overview_map")
    with aside:
        section("Les meilleurs créneaux")
        slot_cards(slots)
        st.caption("Ajustez les volumes et les coûts dans Impact. Préparez vos missions dans Livraisons.")


def show_impact(forecast,axes,scenario,result,day):
    intro("Vos gains, en détail.","Ajustez votre activité : les résultats sont partagés avec la vue d’ensemble.")
    a,b,c=st.columns(3)
    with a: st.number_input("Livraisons par jour",1,5000,step=5,key="scenario_deliveries")
    with b: st.selectbox("Départ habituel",list(range(24)),format_func=lambda hour:f"{hour:02d}:00",key="scenario_baseline")
    with c: st.selectbox("Départ optimisé",list(range(24)),format_func=lambda hour:f"{hour:02d}:00",key="scenario_optimized")
    gain_metrics(result)
    left,right=st.columns([1.4,1],gap="large")
    with left:
        section("Avant / après", "COÛT QUOTIDIEN")
        with st.container(border=True):
            costs=[result["baseline_daily_cost"],result["optimized_daily_cost"]]
            fig=go.Figure(go.Bar(x=["Départ habituel","Départ optimisé"],y=costs,
                marker_color=["#d1dccb","#5d9168"],width=.4,
                text=[f"{fmt(value)} €" for value in costs],textposition="outside",
                hovertemplate="%{x} · %{y:.0f} €<extra></extra>"))
            fig.update_layout(height=260,margin=dict(l=10,r=10,t=30,b=10),
                paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                yaxis=dict(gridcolor="#edf0e9",rangemode="tozero",title="€/jour"),
                font=dict(family="Segoe UI, sans-serif",color="#7c8976",size=11))
            plot(fig,"impact_comparison")
    with right:
        section("Projection annuelle",f"{scenario['working_days']} JOURS COMPARABLES")
        st.markdown(f'<div class="big-impact"><div class="value">{fmt(result["annual_savings"])} €</div>'
            '<div class="caption">Économies estimées à activité et conditions identiques.</div></div>',unsafe_allow_html=True)
        breakdown=[]
        for axe in axes:
            parameters={**scenario,"deliveries":scenario["deliveries"]/len(axes)}
            row=estimate_impact(forecast,[axe],**parameters)
            breakdown.append({"Axe":LABELS[axe],"Gain / jour (€)":round(row["daily_savings"],1),
                "Temps gagné (min)":round(row["time_saved_minutes"],1)})
        st.dataframe(pd.DataFrame(breakdown),hide_index=True,width="stretch")
        st.caption("Répartition uniforme des livraisons entre les axes sélectionnés.")
    with st.expander("Ajuster les coûts et les durées"):
        a,b,c=st.columns(3)
        with a:
            st.number_input("Trajet en circulation fluide (min)",5,120,step=5,key="scenario_travel")
            st.number_input("Service par livraison (min)",1,120,step=1,key="scenario_service")
        with b:
            st.number_input("Coût de l’équipe (€/heure)",0.,200.,step=1.,key="scenario_cost")
            st.number_input("Coût véhicule en trajet (€/heure)",0.,80.,step=1.,key="scenario_vehicle")
        with c:
            st.number_input("Jours comparables par an",1,300,step=1,key="scenario_days")
        st.caption("Trajet simulé = trajet fluide × (1 + occupation / 50). Coût = temps de trajet et de service × coût équipe, plus coût véhicule pendant le trajet. Relation illustrative, non calibrée sur des temps de parcours mesurés. Une valeur négative indique un surcoût. Les gains affichés sont estimés.")
    report=pd.DataFrame([{"date":str(day),"axes":", ".join(axes),**scenario,**result}])
    st.download_button("Exporter la simulation · CSV",report.to_csv(index=False).encode("utf-8-sig"),
        f"flowpilot_impact_{day}.csv","text/csv",key="impact_export")


def show_deliveries(forecast,axes,slots,scenario,day,source_key):
    best=slots.iloc[0]
    intro("Préparez vos livraisons.",f"Créneau recommandé : {int(best.start_hour):02d}h–{int(best.end_hour):02d}h · Ajoutez vos missions, puis construisez le planning.")
    a,b=st.columns([1,3],vertical_alignment="bottom")
    with a: fleet=st.number_input("Véhicules disponibles",1,30,3,step=1,key="fleet")
    with b: st.caption("L’heure limite inclut le trajet et le service. Chaque véhicule effectue une seule mission à la fois.")
    default=pd.DataFrame({"Axe":[LABELS[axes[index%len(axes)]] for index in range(6)],
        "Disponible dès":[8,8,9,10,11,12],"Terminer avant":[12,13,14,16,18,19],
        "Service (min)":[8]*6})
    edited=st.data_editor(default,num_rows="dynamic",hide_index=True,width="stretch",key="delivery_editor_"+"_".join(axes),
        column_config={"Axe":st.column_config.SelectboxColumn("Axe",options=[LABELS[axe] for axe in axes],required=True),
            "Disponible dès":st.column_config.NumberColumn("Disponible dès",min_value=0,max_value=23,step=1,format="%dh",required=True),
            "Terminer avant":st.column_config.NumberColumn("Terminer avant",min_value=1,max_value=24,step=1,format="%dh",required=True),
            "Service (min)":st.column_config.NumberColumn("Service (min)",min_value=1,max_value=120,step=1,required=True)})
    with st.expander("Heures d’exploitation"):
        operating=st.slider("Plage horaire",0,24,(7,20),format="%02dh",key="plan_range")
        st.caption(f"Trajet fluide de référence : {scenario['free_flow_minutes']} minutes, réglable dans Impact. Les trajets sont estimés indépendamment, sans calcul d’itinéraire entre adresses.")
    requests=edited.rename(columns={"Axe":"axe","Disponible dès":"earliest_hour",
        "Terminer avant":"latest_hour","Service (min)":"service_minutes"}).copy()
    requests["axe"]=requests.axe.map({label:axe for axe,label in LABELS.items()})
    signature=hashlib.sha256((source_key+str(day)+requests.to_json()+
        str((fleet,operating,scenario["free_flow_minutes"])) ).encode()).hexdigest()
    if st.button("Construire mon planning",type="primary",disabled=requests.empty or operating[0]>=operating[1],key="build_schedule"):
        schedule=build_schedule(forecast,requests,fleet_size=int(fleet),start_hour=operating[0],
            end_hour=operating[1],base_travel_minutes=scenario["free_flow_minutes"])
        st.session_state["saved_schedule"]={"signature":signature,"data":schedule}
    stored=st.session_state.get("saved_schedule")
    if not stored:
        return
    if stored["signature"]!=signature:
        st.info("Les paramètres ont changé. Reconstruisez le planning pour actualiser la proposition.")
        return
    schedule=stored["data"]
    planned=schedule.status.eq("Planifiée")
    total_minutes,total_cost=0.,0.
    for row in schedule.loc[planned].itertuples():
        duration=(row.end-row.start).total_seconds()/60
        total_minutes+=duration
        total_cost+=(duration*scenario["hourly_cost"]+
            row.travel_minutes*scenario["fuel_cost_per_hour"])/60
    section("Votre feuille de route",f"{schedule.loc[planned,'vehicle'].nunique()} VÉHICULES")
    a,b,c=st.columns(3)
    with a: metric_card("Livraisons planifiées",f"{int(planned.sum())} / {len(schedule)}",note="Dans les disponibilités demandées")
    with b: metric_card("Durée cumulée estimée",fmt(total_minutes/60,1),"h",note="Trajets et service · tous véhicules")
    with c: metric_card("Coût estimé du planning",fmt(total_cost),"€",note="Sur les missions planifiées",featured=True)
    if planned.any():
        plot(schedule_chart(schedule,LABELS),"delivery_schedule")
    if (~planned).any():
        st.warning(f"{int((~planned).sum())} livraison(s) ne peuvent pas être planifiées avec ces contraintes.")
    with st.expander("Détail des missions"):
        view=schedule.copy()
        view["axe"]=view.axe.map(LABELS).fillna("Axe non renseigné")
        for column in ["start","end"]:
            view[column]=view[column].apply(lambda value:pd.Timestamp(value).strftime("%H:%M") if pd.notna(value) else "—")
        st.dataframe(view[["request_id","axe","start","end","vehicle","status","reason"]].rename(columns={
            "request_id":"Livraison","axe":"Axe","start":"Départ","end":"Fin estimée",
            "vehicle":"Véhicule","status":"Statut","reason":"Détail"}),hide_index=True,width="stretch")
    a,b,_=st.columns([1,1,2])
    with a: st.download_button("Télécharger le planning",schedule.to_csv(index=False).encode("utf-8-sig"),
        f"flowpilot_planning_{day}.csv","text/csv",key="planning_export",width="stretch")
    with b: st.download_button("Ajouter à mon agenda",export_calendar(schedule),
        f"flowpilot_livraisons_{day}.ics","text/calendar",key="agenda_export",disabled=not planned.any(),width="stretch")


def main():
    st.markdown('<div class="app-header"><div class="brand"><div class="brand-mark">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><path d="M5 17L11 5h8l-6 12H5Z" stroke="#244e37" stroke-width="2"/>'
        '<path d="M4 21h9M11 17l-2 4" stroke="#244e37" stroke-width="2"/></svg></div>'
        '<div class="brand-name">flow<span>pilot</span></div></div><div class="header-meta">Livrer mieux, simplement. · Paris</div></div>',unsafe_allow_html=True)
    a,b,c=st.columns([1.2,2.8,.7],vertical_alignment="bottom")
    with a: day=st.date_input("Journée à explorer",value=date(2026,11,7),min_value=date(2026,11,7),
        max_value=date(2026,11,11),format="DD/MM/YYYY",key="selected_day")
    with b: axes=st.multiselect("Axes suivis",list(AXES),default=list(AXES),format_func=lambda axe:LABELS[axe],key="selected_axes")
    with c:
        with st.popover("Réglages",width="stretch"):
            source=st.radio("Moteur utilisé",["Profil historique","CSV de l’équipe"],key="forecast_source")
            upload=st.file_uploader("Prévisions au format BCG",type=["csv"],key="forecast_upload") if source=="CSV de l’équipe" else None
            st.caption("CSV : arc, datetime, debit_horaire, taux_occupation. Une journée complète = 72 lignes. Heures de début des créneaux, à Paris.")
            st.caption("Par défaut : profil historique saisonnier, en attendant les prévisions finales de l’équipe. Le capteur Champs-Élysées n’a pas de mesure récente. Occupation et débit ne mesurent pas une densité en véhicules/km.")
    if not axes:
        st.info("Sélectionnez au moins un axe.")
        st.stop()
    imported,source_key=False,"historical"
    try:
        if source=="CSV de l’équipe" and upload is not None:
            content=upload.getvalue()
            frame=imported_data(content)
            forecast=frame[frame.datetime.dt.date.eq(day)].copy()
            actual={(row.axe,row.datetime.hour) for row in forecast.itertuples()}
            if actual!={(axe,hour) for axe in AXES for hour in range(24)}:
                st.error("Le fichier doit contenir les 24 heures des 3 axes pour la journée sélectionnée (72 lignes).")
                st.stop()
            imported=True
            source_key=hashlib.sha256(content).hexdigest()
        else:
            forecast=forecast_for(day)
            if source=="CSV de l’équipe": st.info("Ajoutez votre CSV dans Réglages. Le profil historique est utilisé en attendant.")
    except (FileNotFoundError,ValueError,KeyError,pd.errors.ParserError) as error:
        st.error(f"Impossible de préparer les prévisions : {error}")
        st.stop()
    slots=rank_slots(forecast,axes,7,20,2)
    scenario=init_scenario(slots.iloc[0].start_hour)
    result=estimate_impact(forecast,axes,**scenario)
    overview_tab,impact_tab,delivery_tab=st.tabs(["Vue d’ensemble","Impact","Livraisons"])
    with overview_tab: show_overview(forecast,axes,slots,scenario,result)
    with impact_tab: show_impact(forecast,axes,scenario,result,day)
    with delivery_tab: show_deliveries(forecast,axes,slots,scenario,day,source_key)
    label="Prévisions de l’équipe · Gains estimés" if imported else "Profil historique saisonnier · Gains estimés"
    st.markdown(f'<div class="footnote"><span>{escape(label)}</span><span>{escape(french_day(day))} · Heures de Paris</span></div>',unsafe_allow_html=True)


if __name__=="__main__":
    main()
