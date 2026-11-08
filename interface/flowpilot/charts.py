"""Graphiques et carte vectorielle locale : aucun jeton ni fond distant requis."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
PALETTE = {"Champs-Elysees": "#6d8e63", "Convention": "#b49758", "Saints-Peres": "#628d98"}
LABELS = {"Champs-Elysees": "Champs-Élysées", "Convention": "Convention", "Saints-Peres": "Saints-Pères"}


def traffic_color(occupation: float) -> str:
    return "#5a9871" if occupation < 15 else "#d2a254" if occupation < 30 else "#d47857" if occupation < 50 else "#b65360"


def traffic_label(occupation: float) -> str:
    return "Fluide" if occupation < 15 else "Pré-saturé" if occupation < 30 else "Saturé" if occupation < 50 else "Bloqué"


def _project(lon, lat):
    return (np.asarray(lon) - 2.34) * 73.2, (np.asarray(lat) - 48.857) * 111.2


def paris_map(rows: pd.DataFrame, axes_meta: dict, metric: str = "occupation") -> go.Figure:
    """Carte pan/zoom de Paris. Contours officiels + segments des capteurs réels.

    Le tracé de Seine est un repère schématique ; la carte ne calcule aucun trajet.
    """
    fig = go.Figure()
    context = ROOT / "assets" / "paris_arrondissements.geojson"
    if context.exists():
        features = json.loads(context.read_text(encoding="utf-8"))["features"]
        for f in features:
            geom, prop = f["geometry"], f["properties"]
            polygons = [geom["coordinates"]] if geom["type"] == "Polygon" else geom["coordinates"]
            for polygon in polygons:
                ring = np.asarray(polygon[0])
                x, y = _project(ring[:, 0], ring[:, 1])
                fig.add_trace(go.Scatter(x=x.tolist(), y=y.tolist(), mode="lines", fill="toself",
                    fillcolor="#e9ede5" if prop.get("c_ar") in [8, 15, 7] else "#f1f3ec",
                    line=dict(color="#d9e1d4", width=1), hoverinfo="skip", showlegend=False))
            center = prop.get("geom_x_y", {})
            if center:
                x, y = _project(center["lon"], center["lat"])
                fig.add_trace(go.Scatter(x=[float(x)], y=[float(y)], mode="text", text=[str(prop["c_ar"])],
                    textfont=dict(size=11, color="#b4bfb0"), hoverinfo="skip", showlegend=False))
    # Repère paysager schématique, sans prétendre à une géométrie hydrographique précise.
    seine = np.array([[2.254,48.834],[2.268,48.842],[2.278,48.851],[2.291,48.860],
        [2.305,48.864],[2.320,48.863],[2.336,48.858],[2.351,48.853],[2.366,48.847],[2.384,48.834],[2.407,48.824]])
    sx, sy = _project(seine[:,0], seine[:,1])
    fig.add_trace(go.Scatter(x=sx.tolist(), y=sy.tolist(), mode="lines", line=dict(color="#c8dfe1", width=9, shape="spline"),
        hoverinfo="skip", showlegend=False))
    for lon, lat, label in [(2.2945,48.8584,"TOUR EIFFEL"),(2.3376,48.8606,"LOUVRE"),(2.3522,48.8867,"MONTMARTRE"),(2.399,48.847,"PARIS")]:
        x,y = _project(lon,lat)
        fig.add_trace(go.Scatter(x=[float(x)],y=[float(y)],mode="text",text=[label],textfont=dict(size=8,color="#a3afa1"),hoverinfo="skip",showlegend=False))
    offsets = {"Champs-Elysees":(-1.05,.60),"Convention":(-.45,-.70),"Saints-Peres":(.75,.50)}
    flow_max = max(float(rows.debit.max()), 1) if not rows.empty else 1
    for _, row in rows.iterrows():
        meta = axes_meta[row.axe]
        path = np.asarray(meta["path"])
        color = traffic_color(row.occupation)
        x,y = _project(path[:,0],path[:,1])
        fig.add_trace(go.Scatter(x=x.tolist(),y=y.tolist(),mode="lines",line=dict(color=color,width=8),
            hoverinfo="skip",showlegend=False))
        x,y = _project(meta["lon"],meta["lat"])
        value = f"{row.occupation:.1f} %" if metric=="occupation" else f"{row.debit:,.0f} véh/h".replace(","," ")
        size = 30 if metric=="occupation" else 22+float(row.debit)/flow_max*22
        fig.add_trace(go.Scatter(x=[float(x)],y=[float(y)],mode="markers",marker=dict(size=size+18,color=color,opacity=.12),hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter(x=[float(x)],y=[float(y)],mode="markers",marker=dict(size=size,color=color,line=dict(color="#fff",width=3)),
            customdata=[[row.axe]],name=LABELS[row.axe],showlegend=False,
            hovertemplate=f"<b>{LABELS[row.axe]}</b><br>{row.occupation:.1f} % d’occupation<br>{row.debit:.0f} véh/h<br>{traffic_label(row.occupation)}<extra></extra>"))
        dx,dy=offsets[row.axe]
        fig.add_annotation(x=float(x)+dx,y=float(y)+dy,text=f"<b>{LABELS[row.axe]}</b><br>{value}",
            showarrow=False,font=dict(size=11,color="#3a5140"),bgcolor="rgba(255,255,255,.92)",borderpad=7,bordercolor="#e0e6d9",borderwidth=1)
    fig.update_layout(height=430,margin=dict(l=0,r=0,t=6,b=0),paper_bgcolor="#f7f9f3",plot_bgcolor="#f7f9f3",
        xaxis=dict(visible=False,range=[-6.8,6.5],fixedrange=False),
        yaxis=dict(visible=False,range=[-4.9,5.7],scaleanchor="x",scaleratio=1,fixedrange=False),
        dragmode="pan",font=dict(family="DM Sans, sans-serif"),showlegend=False)
    return fig


def hourly_chart(frame: pd.DataFrame, metric="occupation", selected_hour=None, best_slot=None, band=False) -> go.Figure:
    fig=go.Figure()
    for axe, group in frame.groupby("axe",sort=False):
        g=group.sort_values("datetime")
        x=(g.datetime.dt.hour+g.datetime.dt.minute/60).tolist()
        color=PALETTE.get(axe,"#62866a")
        if band and metric=="occupation":
            fig.add_trace(go.Scatter(x=x+x[::-1],y=g.upper_occupation.tolist()+g.lower_occupation.tolist()[::-1],fill="toself",
                fillcolor="rgba(107,143,100,.10)",line=dict(width=0),hoverinfo="skip",showlegend=False))
        fig.add_trace(go.Scatter(x=x,y=g[metric].tolist(),name=LABELS.get(axe,axe),mode="lines",
            line=dict(color=color,width=2.6,shape="spline"),hovertemplate="%{x:02.0f}h · %{y:.1f}"+(" %" if metric=="occupation" else " véh/h")+"<extra>%{fullData.name}</extra>"))
    if selected_hour is not None:
        fig.add_vline(x=selected_hour,line_color="#7f9680",line_width=1,line_dash="dot")
    if best_slot is not None:
        fig.add_vrect(x0=best_slot[0],x1=best_slot[1],fillcolor="#b5d9a5",opacity=.16,line_width=0,layer="below")
    fig.update_layout(height=260,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif",size=11,color="#7b8978"),hovermode="x unified",
        legend=dict(orientation="h",y=1.17,x=0,font=dict(size=10)),
        xaxis=dict(tickvals=list(range(0,24,3)),ticktext=[f"{h:02d}h" for h in range(0,24,3)],showgrid=False,zeroline=False),
        yaxis=dict(title="Occupation (%)" if metric=="occupation" else "Débit (véh/h)",gridcolor="#e9eee4",zeroline=False,rangemode="tozero"))
    return fig


def schedule_chart(schedule: pd.DataFrame, labels: dict) -> go.Figure:
    fig=go.Figure()
    planned=schedule.loc[schedule.status.eq("Planifiée")]
    for _,r in planned.iterrows():
        start=pd.Timestamp(r.start)
        end=pd.Timestamp(r.end)
        fig.add_trace(go.Bar(x=[(end-start).total_seconds()*1000],base=[start.isoformat()],y=[f"Véhicule {r.vehicle}"],orientation="h",
            marker_color=PALETTE.get(r.axe,"#62866a"),text=[labels.get(r.axe,r.axe)],textposition="inside",insidetextanchor="middle",
            name=labels.get(r.axe,r.axe),showlegend=False,hovertemplate=f"{labels.get(r.axe,r.axe)}<br>{start:%H:%M} → {end:%H:%M}<extra></extra>"))
    fig.update_layout(barmode="overlay",height=max(230,planned.vehicle.nunique()*65+70),margin=dict(l=10,r=10,t=10,b=10),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis=dict(type="date",tickformat="%H:%M",gridcolor="#e9eee4"),
        yaxis=dict(autorange="reversed"),font=dict(family="DM Sans, sans-serif",color="#7b8978",size=11))
    return fig
