import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, datetime
from dateutil.relativedelta import relativedelta

st.set_page_config(page_title="Livrets: intérêts, fiscalité et évolutions", page_icon="💶", layout="wide")

st.title("💶 Visualiseur d'intérêts avec fiscalité, périodes de taux et graphiques")
st.caption("Ajoutez des placements, définissez les périodes de taux dans l’année, la fiscalité, et visualisez brut/net avec courbes d’évolution.")

# =========================
# Helpers & Calculs
# =========================

def mois_range(start: date, end: date):
    """Génère la liste des (année, mois) du 1er du mois entre start et end inclus (bornes mensualisées)."""
    # Normalise sur le 1er du mois
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
    """Renvoie le nombre de jours d'une période [period_start, period_end] qui tombe dans le mois de month_start."""
    month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
    s = max(period_start, month_start)
    e = min(period_end, month_end)
    if e < s:
        return 0
    return (e - s).days + 1

def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()

def compute_brut_net_for_month(capital: float, taux_annuel: float, jours: int, base_jour: int, tax_rate: float):
    """Calcule brut et net pour une portion de mois: intérêts simples, prorata jours/base."""
    interet_brut = capital * (taux_annuel/100.0) * (jours / base_jour)
    interet_net = interet_brut * (1.0 - tax_rate/100.0)
    return interet_brut, interet_net

def build_monthly_schedule(placement: dict, start_global: date, end_global: date, base_jour=365):
    """
    Construit un DataFrame mensuel pour un placement:
    Colonnes: ['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net']
    - periods: liste de dicts {'debut':'YYYY-MM-DD','fin':'YYYY-MM-DD','taux':float}
    - fiscalite: {'type': 'PFU'|'PERSONNALISE', 'taux': float}
    """
    nom = placement['nom']
    capital = float(placement['somme'])
    fiscalite = placement.get('fiscalite', {'type': 'PFU', 'taux': 30.0})
    tax_rate = float(fiscalite.get('taux', 30.0))

    # Périodes de taux
    periods = placement.get('periodes', [])
    if not periods:
        # Fallback: taux fixe unique si aucune période fournie
        taux = float(placement.get('taux', 0.0))
        periods = [{
            'debut': start_global.strftime("%Y-%m-%d"),
            'fin': end_global.strftime("%Y-%m-%d"),
            'taux': taux
        }]

    # Normaliser périodes sur l'intervalle global
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
        return pd.DataFrame(columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net'])

    # Construire calendrier mensuel
    months = mois_range(start_global, end_global)
    rows = []
    for m_start in months:
        total_jours = nb_jours_mois(m_start)
        interet_brut_m = 0.0
        interet_net_m = 0.0
        # On mélange les périodes de taux qui tombent dans ce mois
        # Strat: somme des (intérêts sur sous-période) en prorata jours
        taux_effectif_explicatif = []  # pour info
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

        # Si aucune période n’a recouvert ce mois (gap), intérêts nuls
        taux_affiche = 0.0
        if taux_effectif_explicatif:
            # taux pondéré par jours dans le mois (sur base 30j/31j n’intervient pas, on affiche indicatif)
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
    return df

# =========================
# État & Saisie
# =========================

if 'placements' not in st.session_state:
    st.session_state.placements = []  # liste de dicts placements

if 'periode_globale' not in st.session_state:
    # Par défaut: année en cours
    today = date.today()
    start_default = date(today.year, 1, 1)
    end_default = date(today.year, 12, 31)
    st.session_state.periode_globale = {'debut': start_default, 'fin': end_default}

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
    base_jour = st.number_input("Base jours/an", min_value=360, max_value=366, value=365, step=1, help="365 par défaut. 360 possible pour certaines conventions.")

st.subheader("Ajouter ou modifier un placement")

with st.expander("Ajouter un placement"):
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        nom = st.text_input("Nom du placement", placeholder="Livret A")
    with col2:
        somme = st.number_input("Somme investie (€)", min_value=0.0, step=100.0, format="%.2f")
    with col3:
        taux_defaut = st.number_input("Taux annuel (%) - défaut", min_value=0.0, step=0.05, format="%.3f",
                                      help="Utilisé si aucune période de taux n’est ajoutée.")

    st.markdown("Fiscalité")
    colf1, colf2 = st.columns([1,1])
    with colf1:
        fiscalite_type = st.selectbox("Type de fiscalité", ["PFU (30%)", "Personnalisé"], index=0)
    with colf2:
        fiscalite_taux = st.number_input("Taux fiscal (%)", min_value=0.0, max_value=100.0,
                                         value=30.0 if fiscalite_type.startswith("PFU") else 0.0,
                                         step=0.5, format="%.2f")

    st.markdown("Périodes de taux (facultatif)")
    st.caption("Ajoutez une ou plusieurs périodes avec des taux différents dans l’année.")
    per_list = st.session_state.get('periodes_temp', [])
    if 'periodes_temp' not in st.session_state:
        st.session_state.periodes_temp = []

    cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
    with cp1:
        p_deb = st.date_input("Début", value=st.session_state.periode_globale['debut'], key="p_deb_add")
    with cp2:
        p_fin = st.date_input("Fin", value=st.session_state.periode_globale['fin'], key="p_fin_add")
    with cp3:
        p_taux = st.number_input("Taux (%)", min_value=0.0, step=0.05, format="%.3f", key="p_taux_add")
    with cp4:
        if st.button("Ajouter période"):
            if p_deb <= p_fin:
                st.session_state.periodes_temp.append({
                    'debut': p_deb.strftime("%Y-%m-%d"),
                    'fin': p_fin.strftime("%Y-%m-%d"),
                    'taux': float(p_taux)
                })
            else:
                st.warning("La date de début de période doit précéder la date de fin.")

    # Aperçu des périodes ajoutées
    if st.session_state.periodes_temp:
        st.dataframe(pd.DataFrame(st.session_state.periodes_temp), use_container_width=True, hide_index=True)
        if st.button("Vider les périodes"):
            st.session_state.periodes_temp = []

    if st.button("Enregistrer le placement"):
        if not nom:
            st.warning("Veuillez saisir un nom de placement.")
        else:
            placement = {
                'nom': nom.strip(),
                'somme': somme,
                'taux': taux_defaut,
                'fiscalite': {
                    'type': 'PFU' if fiscalite_type.startswith("PFU") else 'PERSONNALISE',
                    'taux': fiscalite_taux
                },
                'periodes': st.session_state.periodes_temp.copy()
            }
            st.session_state.placements.append(placement)
            st.session_state.periodes_temp = []
            st.success(f"Placement « {placement['nom']} » enregistré.")

# Liste des placements
st.subheader("Placements")
if not st.session_state.placements:
    st.info("Aucun placement pour l’instant. Ajoutez-en via le panneau ci-dessus.")
else:
    # Tableau synthèse des réglages
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

    # Calculs détaillés
    debut = st.session_state.periode_globale['debut']
    fin = st.session_state.periode_globale['fin']
    all_monthly = []
    for p in st.session_state.placements:
        dfp = build_monthly_schedule(p, debut, fin, base_jour=base_jour)
        all_monthly.append(dfp)
    if all_monthly:
        monthly = pd.concat(all_monthly, ignore_index=True)
    else:
        monthly = pd.DataFrame(columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net'])

    # Agrégats
    st.markdown("### Résultats mensuels (brut et net)")
    st.dataframe(
        monthly.sort_values(['Date','Placement']),
        use_container_width=True,
        hide_index=True
    )

    # Totaux par placement
    totals_by_pl = monthly.groupby('Placement', as_index=False).agg({
        'Int_brut': 'sum', 'Int_net': 'sum', 'Capital':'first'
    }).rename(columns={'Int_brut':'Total brut (€)','Int_net':'Total net (€)','Capital':'Capital (€)'})
    totals_by_pl['Total brut (€)'] = totals_by_pl['Total brut (€)'].round(2)
    totals_by_pl['Total net (€)'] = totals_by_pl['Total net (€)'].round(2)

    st.markdown("### Totaux par placement")
    st.dataframe(totals_by_pl, use_container_width=True, hide_index=True)

    colm1, colm2, colm3, colm4 = st.columns(4)
    total_capital = float(totals_by_pl['Capital (€)'].sum()) if not totals_by_pl.empty else 0.0
    total_brut = float(totals_by_pl['Total brut (€)'].sum()) if not totals_by_pl.empty else 0.0
    total_net = float(totals_by_pl['Total net (€)'].sum()) if not totals_by_pl.empty else 0.0
    colm1.metric("Capital total", f"{total_capital:,.2f} €".replace(",", " "))
    colm2.metric("Intérêts bruts totaux", f"{total_brut:,.2f} €".replace(",", " "))
    colm3.metric("Intérêts nets totaux", f"{total_net:,.2f} €".replace(",", " "))
    # Intérêt moyen net (%) sur capital
    avg_net_rate = (total_net / total_capital * 100.0) if total_capital > 0 else 0.0
    colm4.metric("Rendement net moyen", f"{avg_net_rate:.2f} %")

    # =========================
    # Graphiques d’évolution
    # =========================
    st.markdown("## Graphiques d’évolution")
    st.caption("Évolutions mensuelles des intérêts (brut/net), par placement et cumulées.")
    # Préparer séries
    monthly_sorted = monthly.sort_values('Date')
    # Par placement
    st.markdown("### Par placement")
    for nom_pl in monthly_sorted['Placement'].unique():
        dfp = monthly_sorted[monthly_sorted['Placement'] == nom_pl].copy()
        dfp_display = dfp[['Date','Int_brut','Int_net']].set_index('Date')
        st.line_chart(dfp_display, height=220, use_container_width=True)
        # Cumul
        dfp_cum = dfp_display.cumsum()
        st.area_chart(dfp_cum.rename(columns={'Int_brut':'Brut cumulé','Int_net':'Net cumulé'}), height=180, use_container_width=True)

    # Cumul global
    st.markdown("### Cumul global")
    glob = monthly_sorted.groupby('Date', as_index=True)[['Int_brut','Int_net']].sum()
    st.line_chart(glob, height=260, use_container_width=True)
    st.area_chart(glob.cumsum().rename(columns={'Int_brut':'Brut cumulé','Int_net':'Net cumulé'}), height=220, use_container_width=True)

    # =========================
    # Export
    # =========================
    st.markdown("## Export")
    # Export des lignes mensuelles et des totaux
    csv_monthly = monthly.sort_values(['Date','Placement']).copy()
    csv_monthly['Date'] = csv_monthly['Date'].astype(str)
    csv_bytes = csv_monthly.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les résultats mensuels (CSV)", data=csv_bytes, file_name="resultats_mensuels.csv", mime="text/csv")

    csv_totals = totals_by_pl.copy()
    csv_totals_bytes = csv_totals.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les totaux par placement (CSV)", data=csv_totals_bytes, file_name="totaux_par_placement.csv", mime="text/csv")

st.divider()
with st.expander("Notes & limites"):
    st.markdown(
        "- Les intérêts sont calculés en prorata linéaire sur une base jours/an configurable (par défaut 365).\n"
        "- La fiscalité est appliquée comme un pourcentage sur les intérêts (modèle simple). Pour des cas réels (prélèvements sociaux, exonérations, seuils), adapter au besoin.\n"
        "- Les périodes de taux permettent de modéliser les changements de taux en cours d'année. En l’absence de période, le taux défaut s’applique.\n"
        "- Les graphiques affichent les intérêts mensuels et leurs cumuls brut/net.\n"
        "- Option avancée possible: règle des quinzaines (livrets FR) et dates de valeur bancaires."
    )
