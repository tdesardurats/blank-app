import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import io

st.set_page_config(page_title="Livrets: intérêts, fiscalité, périodes & graphes", page_icon="💶", layout="wide")
st.title("💶 Visualiseur d'intérêts avec fiscalité, périodes de taux, tableaux et graphes")
st.caption("Ajoutez des placements, définissez les périodes de taux, la fiscalité, et visualisez brut/net avec tableaux et courbes. Export/Import CSV inclus.")

CSV_HEADER = [
    "type_ligne", "nom", "somme", "taux_defaut", "fisc_type", "fisc_taux",
    "periode_debut", "periode_fin", "periode_taux"
]

def export_inputs_to_csv(placements: list, periode_globale: dict) -> bytes:
    rows = []
    rows.append({
        "type_ligne": "META", "nom": "", "somme": "", "taux_defaut": "", "fisc_type": "", "fisc_taux": "",
        "periode_debut": periode_globale['debut'].strftime("%Y-%m-%d"),
        "periode_fin": periode_globale['fin'].strftime("%Y-%m-%d"),
        "periode_taux": ""
    })
    for p in placements:
        nom, somme, taux_defaut = p.get('nom', ''), p.get('somme', 0.0), p.get('taux', 0.0)
        fisc_type = p.get('fiscalite', {}).get('type', 'PFU')
        fisc_taux = p.get('fiscalite', {}).get('taux', 30.0)
        periods = p.get('periodes', [])
        if not periods:
            rows.append({
                "type_ligne": "PLACEMENT", "nom": nom, "somme": somme,
                "taux_defaut": taux_defaut, "fisc_type": fisc_type, "fisc_taux": fisc_taux,
                "periode_debut": "", "periode_fin": "", "periode_taux": ""
            })
        else:
            for per in periods:
                rows.append({
                    "type_ligne": "PLACEMENT", "nom": nom, "somme": somme,
                    "taux_defaut": taux_defaut, "fisc_type": fisc_type, "fisc_taux": fisc_taux,
                    "periode_debut": per.get('debut', ''), "periode_fin": per.get('fin', ''), "periode_taux": per.get('taux', '')
                })
    df = pd.DataFrame(rows, columns=CSV_HEADER)
    return df.to_csv(index=False).encode("utf-8")

def import_inputs_from_csv(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
    missing = [c for c in CSV_HEADER if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
    meta = df[df['type_ligne'] == 'META']
    if not meta.empty:
        m0 = meta.iloc[0]
        try:
            g_debut = datetime.strptime(m0['periode_debut'], "%Y-%m-%d").date()
            g_fin = datetime.strptime(m0['periode_fin'], "%Y-%m-%d").date()
        except Exception:
            today = date.today()
            g_debut, g_fin = date(today.year,1,1), date(today.year,12,31)
    else:
        today = date.today()
        g_debut, g_fin = date(today.year,1,1), date(today.year,12,31)
    periode_globale = {"debut": g_debut, "fin": g_fin}
    plc_rows = df[df['type_ligne'] == 'PLACEMENT'].copy()
    def to_float_safe(x, default=0.0):
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return default
    placements_dict = {}
    for _, r in plc_rows.iterrows():
        nom = r['nom'].strip()
        if not nom: continue
        if nom not in placements_dict:
            placements_dict[nom] = {
                "nom": nom, "somme": to_float_safe(r['somme'], 0.0), "taux": to_float_safe(r['taux_defaut'], 0.0),
                "fiscalite": {"type": r['fisc_type'] if r['fisc_type'] in ("PFU", "PERSONNALISE") else "PFU",
                              "taux": to_float_safe(r['fisc_taux'], 30.0)},
                "periodes": []
            }
        p_deb, p_fin, p_taux = r['periode_debut'].strip(), r['periode_fin'].strip(), r['periode_taux'].strip()
        if p_deb and p_fin and p_taux:
            placements_dict[nom]["periodes"].append({
                "debut": p_deb,
                "fin": p_fin,
                "taux": to_float_safe(p_taux, placements_dict[nom]["taux"])
            })
    placements = list(placements_dict.values())
    return placements, periode_globale

def mois_range(start: date, end: date):
    cur, last = date(start.year, start.month, 1), date(end.year, end.month, 1)
    res = []
    while cur <= last:
        res.append(cur)
        cur = (cur + relativedelta(months=1))
    return res

def nb_jours_mois(d: date): return (d + relativedelta(months=1) - d).days

def clip_period_to_month(period_start: date, period_end: date, month_start: date) -> int:
    month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
    s, e = max(period_start, month_start), min(period_end, month_end)
    if e < s: return 0
    return (e - s).days + 1

def parse_date(s: str) -> date: return datetime.strptime(s, "%Y-%m-%d").date()

def compute_brut_net_for_month(capital: float, taux_annuel: float, jours: int, base_jour: int, tax_rate: float):
    interet_brut = capital * (taux_annuel/100.0) * (jours / base_jour)
    interet_net = interet_brut * (1.0 - tax_rate/100.0)
    return interet_brut, interet_net

def build_monthly_schedule(placement: dict, start_global: date, end_global: date, base_jour=365):
    nom, capital = placement['nom'], float(placement['somme'])
    fiscalite = placement.get('fiscalite', {'type': 'PFU', 'taux': 30.0})
    tax_rate = float(fiscalite.get('taux', 30.0))
    periods = placement.get('periodes', [])
    if not periods:
        taux = float(placement.get('taux', 0.0))
        periods = [{'debut': start_global.strftime("%Y-%m-%d"), 'fin': end_global.strftime("%Y-%m-%d"), 'taux': taux}]
    norm_periods = []
    for p in periods:
        deb, fin = parse_date(p['debut']), parse_date(p['fin'])
        if fin < start_global or deb > end_global: continue
        deb, fin = max(deb, start_global), min(fin, end_global)
        if deb <= fin: norm_periods.append({'debut': deb, 'fin': fin, 'taux': float(p['taux'])})
    if not norm_periods:
        return pd.DataFrame(columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés', 'Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour'])
    months, rows = mois_range(start_global, end_global), []
    for m_start in months:
        interet_brut_m, interet_net_m = 0.0, 0.0
        taux_effectif_explicatif, jours_pond = [], 0
        for p in norm_periods:
            jours_in_mois = clip_period_to_month(p['debut'], p['fin'], m_start)
            if jours_in_mois <= 0: continue
            ib, in_ = compute_brut_net_for_month(capital, p['taux'], jours_in_mois, base_jour, tax_rate)
            interet_brut_m += ib
            interet_net_m += in_
            taux_effectif_explicatif.append((p['taux'], jours_in_mois))
            jours_pond += jours_in_mois
        taux_affiche = (sum(t*j for t,j in taux_effectif_explicatif)/sum(j for _,j in taux_effectif_explicatif)) if taux_effectif_explicatif else 0.0
        rows.append({
            'Placement': nom, 'Date': m_start, 'Capital': capital, 'Taux(%)': round(taux_affiche, 4),
            'Jours_pondérés': int(jours_pond), 'Int_brut': interet_brut_m, 'Int_net': interet_net_m
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['Int_brut'] = df['Int_brut'].round(2)
        df['Int_net'] = df['Int_net'].round(2)
        df['nb_jours'] = df['Date'].apply(nb_jours_mois)
        df['Brut_moyen_jour'] = (df['Int_brut'] / df['nb_jours']).round(4)
        df['Net_moyen_jour'] = (df['Int_net'] / df['nb_jours']).round(4)
        df['Date'] = pd.to_datetime(df['Date'])
    return df

if 'placements' not in st.session_state: st.session_state.placements = []
if 'periode_globale' not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {'debut': date(today.year, 1, 1), 'fin': date(today.year, 12, 31)}

with st.sidebar:
    st.header("Paramètres globaux")
    col_g1, col_g2 = st.columns(2)
    with col_g1: debut = st.date_input("Début période", value=st.session_state.periode_globale['debut'])
    with col_g2: fin = st.date_input("Fin période", value=st.session_state.periode_globale['fin'])
    if debut > fin: st.error("La date de début doit être avant la date de fin.")
    st.session_state.periode_globale = {'debut': debut, 'fin': fin}
    st.markdown("---")
    st.caption("Base de calcul journalière (prorata linéaire)")
    base_jour = st.number_input("Base jours/an", min_value=360, max_value=366, value=365, step=1, help="365 par défaut. 360 possible selon conventions.")

st.subheader("Ajouter ou modifier un placement")
with st.expander("Ajouter un placement"):
    col1, col2, col3 = st.columns([2,1,1])
    with col1: nom = st.text_input("Nom du placement", placeholder="Livret A")
    with col2: somme = st.number_input("Somme investie (€)", min_value=0.0, step=100.0, format="%.2f")
    with col3: taux_defaut = st.number_input("Taux annuel (%) - défaut", min_value=0.0, step=0.05, format="%.3f", help="Utilisé si aucune période de taux n’est ajoutée.")
    st.markdown("Fiscalité")
    colf1, colf2 = st.columns([1,1])
    with colf1: fiscalite_type = st.selectbox("Type de fiscalité", ["PFU (30%)", "Personnalisé"], index=0)
    with colf2: fiscalite_taux = st.number_input("Taux fiscal (%)", min_value=0.0, max_value=100.0, value=30.0 if fiscalite_type.startswith("PFU") else 0.0, step=0.5, format="%.2f")
    st.markdown("Périodes de taux (facultatif)")
    st.caption("Ajoutez une ou plusieurs périodes avec des taux différents dans l’année.")
    if 'periodes_temp' not in st.session_state: st.session_state.periodes_temp = []
    cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
    with cp1: p_deb = st.date_input("Début", value=st.session_state.periode_globale['debut'], key="p_deb_add")
    with cp2: p_fin = st.date_input("Fin", value=st.session_state.periode_globale['fin'], key="p_fin_add")
    with cp3: p_taux = st.number_input("Taux (%)", min_value=0.0, step=0.05, format="%.3f", key="p_taux_add")
    with cp4:
        if st.button("Ajouter période"):
            if p_deb <= p_fin:
                st.session_state.periodes_temp.append({
                    'debut': p_deb.strftime("%Y-%m-%d"), 'fin': p_fin.strftime("%Y-%m-%d"), 'taux': float(p_taux)
                })
            else:
                st.warning("La date de début de période doit précéder la date de fin.")
    if st.session_state.periodes_temp:
        st.dataframe(pd.DataFrame(st.session_state.periodes_temp), use_container_width=True, hide_index=True)
        if st.button("Vider les périodes"): st.session_state.periodes_temp = []
    if st.button("Enregistrer le placement"):
        if not nom:
            st.warning("Veuillez saisir un nom de placement.")
        else:
            placement = {
                'nom': nom.strip(), 'somme': somme, 'taux': taux_defaut,
                'fiscalite': {'type': 'PFU' if fiscalite_type.startswith("PFU") else 'PERSONNALISE', 'taux': fiscalite_taux},
                'periodes': st.session_state.periodes_temp.copy()
            }
            st.session_state.placements.append(placement)
            st.session_state.periodes_temp = []
            st.success(f"Placement « {placement['nom']} » enregistré.")

st.subheader("Placements")
if not st.session_state.placements:
    st.info("Aucun placement pour l’instant. Ajoutez-en via le panneau ci-dessus.")
else:
    synth_rows = []
    for p in st.session_state.placements:
        synth_rows.append({'Nom': p['nom'], 'Somme (€)': p['somme'], 'Taux défaut (%)': p['taux'], 'Fiscalité (%)': p['fiscalite']['taux'], 'Nb périodes': len(p.get('periodes', []))})
    st.dataframe(pd.DataFrame(synth_rows), use_container_width=True, hide_index=True)
    debut, fin = st.session_state.periode_globale['debut'], st.session_state.periode_globale['fin']
    all_monthly = [build_monthly_schedule(p, debut, fin, base_jour=base_jour) for p in st.session_state.placements]
    monthly = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame(
        columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour']
    )
    if not monthly.empty:
        monthly['Date'] = pd.to_datetime(monthly['Date'], errors='coerce')
    st.markdown("### Résultats mensuels enrichis (brut/net + nb_jours + moyennes/jour)")
    monthly_display = monthly.sort_values(['Date','Placement']).copy()
    if not monthly_display.empty:
        monthly_display['Date'] = pd.to_datetime(monthly_display['Date'], errors='coerce')
        monthly_display_fmt = monthly_display.copy()
        monthly_display_fmt['Date'] = monthly_display_fmt['Date'].dt.strftime("%Y-%m")
        st.dataframe(
            monthly_display_fmt[['Date','Placement','Capital','Taux(%)','nb_jours','Jours_pondérés','Int_brut','Int_net','Brut_moyen_jour','Net_moyen_jour']],
            use_container_width=True, hide_index=True
        )
    else:
        st.info("Aucune donnée mensuelle sur la période sélectionnée.")

    st.markdown("### Totaux par placement")
    if not monthly.empty:
        totals_by_pl = monthly.groupby('Placement', as_index=False).agg({
            'Int_brut': 'sum', 'Int_net': 'sum', 'Capital':'first'
        }).rename(columns={'Int_brut':'Total brut (€)','Int_net':'Total net (€)','Capital':'Capital (€)'})
        totals_by_pl['Total brut (€)'] = totals_by_pl['Total brut (€)'].round(2)
        totals_by_pl['Total net (€)'] = totals_by_pl['Total net (€)'].round(2)
        st.dataframe(totals_by_pl, use_container_width=True, hide_index=True)
        colm1, colm2, colm3, colm4 = st.columns(4)
        total_capital = float(totals_by_pl['Capital (€)'].sum())
        total_brut = float(totals_by_pl['Total brut (€)'].sum())
        total_net = float(totals_by_pl['Total net (€)'].sum())
        colm1.metric("Capital total", f"{total_capital:,.2f} €".replace(",", " "))
        colm2.metric("Intérêts bruts totaux", f"{total_brut:,.2f} €".replace(",", " "))
        colm3.metric("Intérêts nets totaux", f"{total_net:,.2f} €".replace(",", " "))
        avg_net_rate = (total_net / total_capital * 100.0) if total_capital > 0 else 0.0
        colm4.metric("Rendement net moyen", f"{avg_net_rate:.2f} %")
    else:
        totals_by_pl = pd.DataFrame(columns=['Placement','Capital (€)','Total brut (€)','Total net (€)'])

    st.markdown("### Classement par rentabilité nette")
    if not monthly.empty and not totals_by_pl.empty:
        ranked = totals_by_pl.copy()
        ranked['Rendement net (%)'] = ranked.apply(
            lambda r: (r['Total net (€)'] / r['Capital (€)'] * 100.0) if r['Capital (€)'] > 0 else 0.0,
            axis=1
        ).round(2)
        total_net_all = ranked['Total net (€)'].sum()
        ranked['Part du net total (%)'] = ranked['Total net (€)'].apply(
            lambda x: (x / total_net_all * 100.0) if total_net_all > 0 else 0.0
        ).round(2)
        ranked = ranked.sort_values('Total net (€)', ascending=False).reset_index(drop=True)
        ranked['Rang'] = None
        current_rank = 0
        last_value = None
        for i, v in enumerate(ranked['Total net (€)'].tolist()):
            if last_value is None or v < last_value:
                current_rank = i + 1
                last_value = v
            ranked.at[i, 'Rang'] = current_rank
        display_cols = ['Rang', 'Placement', 'Capital (€)', 'Total brut (€)', 'Total net (€)', 'Rendement net (%)', 'Part du net total (%)']
        st.dataframe(ranked[display_cols], use_container_width=True, hide_index=True)
    else:
        st.caption("Classement indisponible: pas de données calculées sur la période.")

    st.markdown("## Graphiques d’évolution (Plotly)")
    monthly_sorted = monthly.sort_values('Date').copy()
    if not monthly_sorted.empty:
        monthly_sorted['Date'] = pd.to_datetime(monthly_sorted['Date'], errors='coerce')
        placements_list = monthly_sorted['Placement'].unique().tolist()
        color_palette = px.colors.qualitative.Safe
        color_map = {plc: color_palette[i % len(color_palette)] for i, plc in enumerate(placements_list)}

        st.markdown("### Par placement - Intérêt mensuel (Brut vs Net)")
        for nom_pl in placements_list:
            dfp = monthly_sorted[monthly_sorted['Placement'] == nom_pl].copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dfp['Date'], y=dfp['Int_brut'], name="Brut", marker_color=color_map[nom_pl],
                                 hovertemplate="Mois: %{x|%Y-%m}<br>Brut: %{y:.2f} €<extra></extra>"))
            fig.add_trace(go.Bar(x=dfp['Date'], y=dfp['Int_net'], name="Net", marker_color="rgba(0,0,0,0.45)",
                                 hovertemplate="Mois: %{x|%Y-%m}<br>Net: %{y:.2f} €<extra></extra>"))
            fig.update_layout(
                barmode='group', title_text=f"{nom_pl} - Intérêts mensuels", legend_title_text="Type",
                xaxis_title="Mois", yaxis_title="€", margin=dict(t=50, l=40, r=20, b=40),
                height=360, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Par placement - Cumul des intérêts (Net)")
        for nom_pl in placements_list:
            dfp = monthly_sorted[monthly_sorted['Placement'] == nom_pl].copy().sort_values('Date')
            dfp['Net_cumulé'] = dfp['Int_net'].cumsum()
            fig = px.line(
                dfp, x='Date', y='Net_cumulé', title=f"{nom_pl} - Net cumulé",
                color_discrete_sequence=[color_map[nom_pl]], labels={'Date': 'Mois', 'Net_cumulé':'€'}
            )
            fig.update_traces(mode='lines+markers', hovertemplate="Mois: %{x|%Y-%m}<br>Net cumulé: %{y:.2f} €<extra></extra>")
            fig.update_layout(margin=dict(t=50, l=40, r=20, b=40), height=320, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cumul annuel par placement (couleur par placement)")
        cum_by_pl = monthly_sorted.copy().sort_values(['Placement','Date'])
        cum_by_pl['
