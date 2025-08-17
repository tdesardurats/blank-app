import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import io
import uuid

# =========================
# Config page
# =========================
st.set_page_config(page_title="Livrets: intérêts, fiscalité, périodes & graphes", page_icon="💶", layout="wide")
st.title("💶 Visualiseur d'intérêts avec fiscalité, périodes de taux, tableaux et graphes")
st.caption("Modifiez tout directement dans les tableaux. Ajout/Suppression de lignes, import/export CSV, graphiques et totaux.")

# =========================
# Utilitaires Export/Import des entrées
# =========================
CSV_HEADER = [
    "type_ligne", "nom", "somme", "taux_defaut", "fisc_type", "fisc_taux",
    "periode_debut", "periode_fin", "periode_taux"
]

def export_inputs_to_csv(placements: list, periode_globale: dict) -> bytes:
    rows = []
    # META
    rows.append({
        "type_ligne": "META",
        "nom": "",
        "somme": "",
        "taux_defaut": "",
        "fisc_type": "",
        "fisc_taux": "",
        "periode_debut": periode_globale['debut'].strftime("%Y-%m-%d"),
        "periode_fin": periode_globale['fin'].strftime("%Y-%m-%d"),
        "periode_taux": ""
    })
    # PLACEMENTS
    for p in placements:
        nom = p.get('nom', '')
        somme = p.get('somme', 0.0)
        taux_defaut = p.get('taux', 0.0)
        fisc_type = p.get('fiscalite', {}).get('type', 'PFU')
        fisc_taux = p.get('fiscalite', {}).get('taux', 30.0)
        periods = p.get('periodes', [])
        if not periods:
            rows.append({
                "type_ligne": "PLACEMENT",
                "nom": nom,
                "somme": somme,
                "taux_defaut": taux_defaut,
                "fisc_type": fisc_type,
                "fisc_taux": fisc_taux,
                "periode_debut": "",
                "periode_fin": "",
                "periode_taux": ""
            })
        else:
            for per in periods:
                rows.append({
                    "type_ligne": "PLACEMENT",
                    "nom": nom,
                    "somme": somme,
                    "taux_defaut": taux_defaut,
                    "fisc_type": fisc_type,
                    "fisc_taux": fisc_taux,
                    "periode_debut": per.get('debut', ''),
                    "periode_fin": per.get('fin', ''),
                    "periode_taux": per.get('taux', '')
                })
    df = pd.DataFrame(rows, columns=CSV_HEADER)
    return df.to_csv(index=False).encode("utf-8")

def import_inputs_from_csv(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
    missing = [c for c in CSV_HEADER if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # Période globale
    meta = df[df['type_ligne'] == 'META']
    if not meta.empty:
        m0 = meta.iloc[0]
        try:
            g_debut = datetime.strptime(m0['periode_debut'], "%Y-%m-%d").date()
            g_fin = datetime.strptime(m0['periode_fin'], "%Y-%m-%d").date()
        except Exception:
            today = date.today()
            g_debut = date(today.year, 1, 1)
            g_fin = date(today.year, 12, 31)
    else:
        today = date.today()
        g_debut = date(today.year, 1, 1)
        g_fin = date(today.year, 12, 31)
    periode_globale = {"debut": g_debut, "fin": g_fin}

    # Placements
    plc_rows = df[df['type_ligne'] == 'PLACEMENT'].copy()

    def to_float_safe(x, default=0.0):
        try:
            return float(str(x).replace(",", "."))
        except Exception:
            return default

    placements_dict = {}
    for _, r in plc_rows.iterrows():
        nom = r['nom'].strip()
        if not nom:
            continue
        if nom not in placements_dict:
            placements_dict[nom] = {
                "uid": str(uuid.uuid4()),
                "nom": nom,
                "somme": to_float_safe(r['somme'], 0.0),
                "taux": to_float_safe(r['taux_defaut'], 0.0),
                "fiscalite": {
                    "type": r['fisc_type'] if r['fisc_type'] in ("PFU", "PERSONNALISE") else "PFU",
                    "taux": to_float_safe(r['fisc_taux'], 30.0)
                },
                "periodes": []
            }
        p_deb = r['periode_debut'].strip()
        p_fin = r['periode_fin'].strip()
        p_taux = r['periode_taux'].strip()
        if p_deb and p_fin and p_taux:
            placements_dict[nom]["periodes"].append({
                "debut": p_deb,
                "fin": p_fin,
                "taux": to_float_safe(p_taux, placements_dict[nom]["taux"])
            })

    placements = list(placements_dict.values())
    return placements, periode_globale

# =========================
# Fonctions calcul
# =========================
def mois_range(start: date, end: date):
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    res = []
    while cur <= last:
        res.append(cur)
        cur = (cur + relativedelta(months=1))
    return res

def nb_jours_mois(d: date):
    nxt = d + relativedelta(months=1)
    return (nxt - d).days

def clip_period_to_month(period_start: date, period_end: date, month_start: date) -> int:
    month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
    s = max(period_start, month_start)
    e = min(period_end, month_end)
    if e < s:
        return 0
    return (e - s).days + 1

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def compute_brut_net_for_month(capital: float, taux_annuel: float, jours: int, base_jour: int, tax_rate: float):
    interet_brut = capital * (taux_annuel/100.0) * (jours / base_jour)
    interet_net = interet_brut * (1.0 - tax_rate/100.0)
    return interet_brut, interet_net

def build_monthly_schedule(placement: dict, start_global: date, end_global: date, base_jour=365):
    nom = placement['nom']
    capital = float(placement['somme'])
    fiscalite = placement.get('fiscalite', {'type': 'PFU', 'taux': 30.0})
    tax_rate = float(fiscalite.get('taux', 30.0))
    periods = placement.get('periodes', [])

    if not periods:
        taux = float(placement.get('taux', 0.0))
        periods = [{
            'debut': start_global.strftime("%Y-%m-%d"),
            'fin': end_global.strftime("%Y-%m-%d"),
            'taux': taux
        }]

    norm_periods = []
    for p in periods:
        deb = parse_date(p['debut'])
        fin = parse_date(p['fin'])
        if fin < start_global or deb > end_global:
            continue
        deb = max(deb, start_global)
        fin = min(fin, end_global)
        if deb <= fin:
            norm_periods.append({'debut': deb, 'fin': fin, 'taux': float(p['taux'])})

    if not norm_periods:
        return pd.DataFrame(columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour'])

    months = mois_range(start_global, end_global)
    rows = []
    for m_start in months:
        interet_brut_m = 0.0
        interet_net_m = 0.0
        taux_effectif_explicatif = []
        jours_pond = 0
        for p in norm_periods:
            jours_in_mois = clip_period_to_month(p['debut'], p['fin'], m_start)
            if jours_in_mois <= 0:
                continue
            ib, in_ = compute_brut_net_for_month(capital, p['taux'], jours_in_mois, base_jour, tax_rate)
            interet_brut_m += ib
            interet_net_m += in_
            taux_effectif_explicatif.append((p['taux'], jours_in_mois))
            jours_pond += jours_in_mois

        taux_affiche = 0.0
        if taux_effectif_explicatif:
            somme_taux_jours = sum(t * j for t, j in taux_effectif_explicatif)
            taux_affiche = somme_taux_jours / sum(j for _, j in taux_effectif_explicatif)

        rows.append({
            'Placement': nom,
            'Date': m_start,
            'Capital': capital,
            'Taux(%)': round(taux_affiche, 4),
            'Jours_pondérés': int(jours_pond),
            'Int_brut': interet_brut_m,
            'Int_net': interet_net_m
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

# =========================
# État
# =========================
if 'placements' not in st.session_state:
    st.session_state.placements = []  # liste de dicts avec uid
if 'periode_globale' not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {'debut': date(today.year, 1, 1), 'fin': date(today.year, 12, 31)}

# Assurer un uid à chaque placement existant
for p in st.session_state.placements:
    if "uid" not in p or not p["uid"]:
        p["uid"] = str(uuid.uuid4())

# =========================
# Barre latérale: paramètres globaux
# =========================
with st.sidebar:
    st.header("Paramètres globaux")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        debut = st.date_input("Début période", value=st.session_state.periode_globale['debut'])
    with col_g2:
        fin = st.date_input("Fin période", value=st.session_state.periode_globale['fin'])
    if debut > fin:
        st.error("La date de début doit être avant la date de fin.")
    st.session_state.periode_globale = {'debut': debut, 'fin': fin}

    st.markdown("---")
    st.caption("Base de calcul journalière (prorata linéaire)")
    base_jour = st.number_input("Base jours/an", min_value=360, max_value=366, value=365, step=1)

# =========================
# Tableaux éditables DIRECTS
# =========================

st.subheader("Placements (éditer directement dans le tableau)")
# Construire DataFrame placements (inclut uid caché si on veut)
pl_df = pd.DataFrame([{
    "uid": p["uid"],
    "Nom": p["nom"],
    "Somme (€)": float(p["somme"]),
    "Taux défaut (%)": float(p["taux"]),
    "Type Fiscalité": p["fiscalite"]["type"],
    "Fiscalité (%)": float(p["fiscalite"]["taux"]),
} for p in st.session_state.placements])

# Stocker la table d'entrée pour suivi
st.session_state["pl_df_source"] = pl_df.copy()

# Affichage éditable
show_uid = st.checkbox("Afficher uid placements (pour lier les périodes)", value=False)
pl_editor = st.data_editor(
    pl_df,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "uid": st.column_config.Column("uid", disabled=not show_uid),
        "Nom": st.column_config.TextColumn("Nom", required=True),
        "Type Fiscalité": st.column_config.SelectboxColumn("Type Fiscalité", options=["PFU", "PERSONNALISE"], required=True),
        "Somme (€)": st.column_config.NumberColumn("Somme (€)", step=50.0, format="%.2f"),
        "Taux défaut (%)": st.column_config.NumberColumn("Taux défaut (%)", step=0.05, format="%.3f"),
        "Fiscalité (%)": st.column_config.NumberColumn("Fiscalité (%)", step=0.5, format="%.2f"),
    },
    key="pl_table",
)

st.subheader("Périodes (éditer directement dans le tableau)")
# Construire DataFrame périodes: une ligne = une période, avec uid_placement
period_rows = []
for p in st.session_state.placements:
    for per in p.get("periodes", []):
        period_rows.append({
            "uid_placement": p["uid"],
            "Début": per["debut"],
            "Fin": per["fin"],
            "Taux (%)": float(per["taux"]),
        })
per_df = pd.DataFrame(period_rows) if period_rows else pd.DataFrame(columns=["uid_placement","Début","Fin","Taux (%)"])
st.session_state["per_df_source"] = per_df.copy()

per_editor = st.data_editor(
    per_df,
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "uid_placement": st.column_config.TextColumn("uid_placement", help="Collez l'uid du placement ciblé"),
        "Début": st.column_config.TextColumn("Début", help="YYYY-MM-DD"),
        "Fin": st.column_config.TextColumn("Fin", help="YYYY-MM-DD"),
        "Taux (%)": st.column_config.NumberColumn("Taux (%)", step=0.05, format="%.3f"),
    },
    key="per_table",
)

def _to_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def _valid_date(s):
    try:
        datetime.strptime(str(s), "%Y-%m-%d")
        return True
    except Exception:
        return False

def apply_all_changes():
    # Appliquer le tableau Placements
    updated_pl = []
    used_uids = set()
    for _, r in st.session_state["pl_table"].iterrows():
        nom = str(r.get("Nom") or "").strip()
        if not nom:
            # ignorer lignes vides
            continue
        uid = str(r.get("uid") or "").strip()
        if not uid:
            uid = str(uuid.uuid4())
        if uid in used_uids:
            # éviter doublons uid dans le tableau
            uid = str(uuid.uuid4())
        used_uids.add(uid)
        fisc_type = str(r.get("Type Fiscalité") or "PFU").strip()
        if fisc_type not in ("PFU", "PERSONNALISE"):
            fisc_type = "PFU"
        updated_pl.append({
            "uid": uid,
            "nom": nom,
            "somme": _to_float(r.get("Somme (€)"), 0.0),
            "taux": _to_float(r.get("Taux défaut (%)"), 0.0),
            "fiscalite": {"type": fisc_type, "taux": _to_float(r.get("Fiscalité (%)"), 30.0)},
            "periodes": []  # temporaire, on réinjecte après depuis per_table
        })

    # Index rapide uid -> placement dict
    uid_to_pl = {p["uid"]: p for p in updated_pl}

    # Appliquer le tableau Périodes
    for _, r in st.session_state["per_table"].iterrows():
        uid_p = str(r.get("uid_placement") or "").strip()
        deb = str(r.get("Début") or "").strip()
        fin = str(r.get("Fin") or "").strip()
        tx = _to_float(r.get("Taux (%)"), None)
        if not uid_p or not deb or not fin or tx is None:
            continue
        if not _valid_date(deb) or not _valid_date(fin):
            continue
        if uid_p not in uid_to_pl:
            # uid inconnu -> ignorer la ligne
            continue
        uid_to_pl[uid_p]["periodes"].append({
            "debut": deb,
            "fin": fin,
            "taux": float(tx)
        })

    st.session_state.placements = list(uid_to_pl.values())
    st.success("Modifications appliquées.")
    # pas de st.rerun ici: on laisse l’utilisateur continuer à éditer; les calculs s’actualisent ci-dessous

st.button("Appliquer modifications", on_click=apply_all_changes, use_container_width=True)

# =========================
# Ajout rapide d’un placement (optionnel)
# =========================
with st.expander("Ajout rapide (facultatif)"):
    ncol1, ncol2 = st.columns([3,1])
    with ncol1:
        new_name = st.text_input("Nom", placeholder="Nouveau livret")
    with ncol2:
        if st.button("Ajouter une ligne vide au tableau Placements"):
            # ajoute une ligne vide dans l'éditeur (via manipulation de son DataFrame en mémoire)
            tmp = st.session_state["pl_table"].copy()
            tmp.loc[len(tmp)] = {"uid": str(uuid.uuid4()), "Nom": new_name, "Somme (€)": 0.0, "Taux défaut (%)": 0.0, "Type Fiscalité": "PFU", "Fiscalité (%)": 30.0}
            st.session_state["pl_table"] = tmp
            st.rerun()

# =========================
# Calculs et affichages
# =========================
st.subheader("Placements (synthèse)")
if not st.session_state.placements:
    st.info("Aucun placement. Ajoutez/modifiez via le tableau ci-dessus puis cliquez sur Appliquer modifications.")
else:
    synth_rows = []
    for p in st.session_state.placements:
        synth_rows.append({
            'Nom': p['nom'],
            'Somme (€)': p['somme'],
            'Taux défaut (%)': p['taux'],
            'Fiscalité (%)': p['fiscalite']['taux'],
            'Nb périodes': len(p.get('periodes', []))
        })
    st.dataframe(pd.DataFrame(synth_rows), use_container_width=True, hide_index=True)

    debut = st.session_state.periode_globale['debut']
    fin = st.session_state.periode_globale['fin']

    # Planning mensuel pour tous
    all_monthly = []
    for p in st.session_state.placements:
        dfp = build_monthly_schedule(p, debut, fin, base_jour=base_jour)
        all_monthly.append(dfp)
    monthly = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame(
        columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour']
    )

    if not monthly.empty:
        monthly['Date'] = pd.to_datetime(monthly['Date'], errors='coerce')

    st.markdown("### Résultats mensuels (brut/net + nb_jours + moyennes/jour)")
    monthly_display = monthly.sort_values(['Date','Placement']).copy()
    if not monthly_display.empty:
        monthly_display['Date'] = pd.to_datetime(monthly_display['Date'], errors='coerce')
        monthly_display_fmt = monthly_display.copy()
        monthly_display_fmt['Date'] = monthly_display_fmt['Date'].dt.strftime("%Y-%m")
        st.dataframe(
            monthly_display_fmt[['Date','Placement','Capital','Taux(%)','nb_jours','Jours_pondérés','Int_brut','Int_net','Brut_moyen_jour','Net_moyen_jour']],
            use_container_width=True,
            hide_index=True
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

    # Graphiques
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
            fig.add_trace(go.Bar(
                x=dfp['Date'], y=dfp['Int_brut'],
                name="Brut",
                marker_color=color_map[nom_pl],
                hovertemplate="Mois: %{x|%Y-%m}<br>Brut: %{y:.2f} €<extra></extra>"
            ))
            fig.add_trace(go.Bar(
                x=dfp['Date'], y=dfp['Int_net'],
                name="Net",
                marker_color="rgba(0,0,0,0.45)",
                hovertemplate="Mois: %{x|%Y-%m}<br>Net: %{y:.2f} €<extra></extra>"
            ))
            fig.update_layout(
                barmode='group',
                title_text=f"{nom_pl} - Intérêts mensuels",
                legend_title_text="Type",
                xaxis_title="Mois",
                yaxis_title="€",
                margin=dict(t=50, l=40, r=20, b=40),
                height=360,
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Par placement - Cumul des intérêts (Net)")
        for nom_pl in placements_list:
            dfp = monthly_sorted[monthly_sorted['Placement'] == nom_pl].copy().sort_values('Date')
            dfp['Net_cumulé'] = dfp['Int_net'].cumsum()
            fig = px.line(
                dfp,
                x='Date', y='Net_cumulé',
                title=f"{nom_pl} - Net cumulé",
                color_discrete_sequence=[color_map[nom_pl]],
                labels={'Date': 'Mois', 'Net_cumulé':'€'}
            )
            fig.update_traces(mode='lines+markers', hovertemplate="Mois: %{x|%Y-%m}<br>Net cumulé: %{y:.2f} €<extra></extra>")
            fig.update_layout(margin=dict(t=50, l=40, r=20, b=40), height=320, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cumul net global (aire empilée)")
        stacked = monthly_sorted.copy()
        stacked['Date_str'] = stacked['Date'].dt.strftime("%Y-%m")
        pivot = stacked.pivot_table(index='Date_str', columns='Placement', values='Int_net', aggfunc='sum').fillna(0)
        pivot_cum = pivot.cumsum()
        fig_stack = go.Figure()
        for nom_pl in placements_list:
            fig_stack.add_trace(go.Scatter(
                x=pivot_cum.index, y=pivot_cum[nom_pl],
                mode='lines',
                name=nom_pl,
                line=dict(color=color_map[nom_pl], width=0.8),
                stackgroup='one',
                hovertemplate="Mois: %{x}<br>Cumul net: %{y:.2f} €<extra></extra>"
            ))
        fig_stack.update_layout(
            title_text="Cumul net global (aire empilée)",
            xaxis_title="Mois",
            yaxis_title="€",
            legend_title_text="Placement",
            margin=dict(t=50, l=40, r=20, b=40),
            height=420,
            template="plotly_white"
        )
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Aucune donnée pour tracer les graphiques sur la période choisie.")

    # =========================
    # Export résultats
    # =========================
    st.markdown("## Export des résultats")
    if not monthly.empty:
        csv_monthly = monthly.sort_values(['Date','Placement']).copy()
        csv_monthly['Date'] = pd.to_datetime(csv_monthly['Date'], errors='coerce').dt.strftime("%Y-%m-%d")
        csv_bytes = csv_monthly.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Télécharger les résultats mensuels (CSV)", data=csv_bytes, file_name="resultats_mensuels.csv", mime="text/csv")

        csv_totals = totals_by_pl.copy()
        csv_totals_bytes = csv_totals.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Télécharger les totaux par placement (CSV)", data=csv_totals_bytes, file_name="totaux_par_placement.csv", mime="text/csv")
    else:
        st.caption("Exports désactivés: aucune donnée calculée.")

# =========================
# Export / Import des données d’entrée
# =========================
st.divider()
st.markdown("## Export / Import des données d’entrée")
col_exp, col_imp = st.columns(2)
with col_exp:
    if st.button("📤 Exporter les données d’entrée (CSV)"):
        csv_in_bytes = export_inputs_to_csv(
            st.session_state.placements,
            st.session_state.periode_globale
        )
        st.download_button(
            "Télécharger le CSV des données d’entrée",
            data=csv_in_bytes,
            file_name="donnees_entree_livrets.csv",
            mime="text/csv",
            use_container_width=True
        )

with col_imp:
    uploaded = st.file_uploader("📥 Importer un CSV de données d’entrée", type=["csv"])
    if uploaded is not None:
        try:
            placements_imp, periode_imp = import_inputs_from_csv(uploaded.read())
            # Garantir uid pour chaque placement importé
            for p in placements_imp:
                p.setdefault("uid", str(uuid.uuid4()))
            st.session_state.placements = placements_imp
            st.session_state.periode_globale = periode_imp
            st.success("Données d’entrée importées. Éditez-les dans les tableaux puis cliquez sur Appliquer modifications.")
        except Exception as e:
            st.error(f"Erreur lors de l’import: {e}")

# =========================
# Notes
# =========================
st.divider()
with st.expander("Notes & limites"):
    st.markdown(
        "- Éditez toutes les cellules directement dans les tableaux. Utilisez « Appliquer modifications » pour persister dans l’état.\n"
        "- La table Périodes relie chaque ligne à un placement via « uid_placement ». Affichez les uid des placements pour les copier/coller facilement.\n"
        "- Intérêts calculés en prorata linéaire (base jours/an configurable). Fiscalité simple (pour cas réels, adapter PFU/PS selon vos besoins).\n"
        "- Graphiques: barres brutes/nettes mensuelles, courbes cumulées par placement et globales empilées.\n"
    )
