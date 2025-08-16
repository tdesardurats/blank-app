import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import io

# =========================
# Config page
# =========================
st.set_page_config(page_title="Livrets: intérêts, fiscalité, périodes & graphes", page_icon="💶", layout="wide")

st.title("💶 Visualiseur d'intérêts avec fiscalité, périodes de taux, tableaux et graphes")
st.caption("Ajoutez des placements, définissez les périodes de taux, la fiscalité, et visualisez brut/net avec tableaux et courbes. Export/Import CSV inclus.")

# =========================
# Utilitaires Export/Import des entrées
# =========================
CSV_HEADER = [
    "type_ligne",          # META ou PLACEMENT
    "nom",                 # placement name
    "somme",               # capital
    "taux_defaut",         # default annual rate
    "fisc_type",           # PFU|PERSONNALISE
    "fisc_taux",           # tax rate %
    "periode_debut",       # global or period start (YYYY-MM-DD)
    "periode_fin",         # global or period end (YYYY-MM-DD)
    "periode_taux"         # period rate (if any)
]

def export_inputs_to_csv(placements: list, periode_globale: dict) -> bytes:
    rows = []
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
        # Ajoute nb_jours du mois et moyennes/jour
        df['nb_jours'] = df['Date'].apply(nb_jours_mois)
        df['Brut_moyen_jour'] = (df['Int_brut'] / df['nb_jours']).round(4)
        df['Net_moyen_jour'] = (df['Int_net'] / df['nb_jours']).round(4)
        # Cast en datetime pour robustesse
        df['Date'] = pd.to_datetime(df['Date'])
    return df

# =========================
# État
# =========================
if 'placements' not in st.session_state:
    st.session_state.placements = []

if 'periode_globale' not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {'debut': date(today.year, 1, 1), 'fin': date(today.year, 12, 31)}

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
    base_jour = st.number_input("Base jours/an", min_value=360, max_value=366, value=365, step=1, help="365 par défaut. 360 possible selon conventions.")

# =========================
# Saisie des placements
# =========================
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

# ======= Ajouts: Edition interactive & bascule conditions ==========

if st.session_state.placements:
    st.markdown("## Modification et analyse avancée des placements")
    df_inputs = pd.DataFrame([
        {
            "Nom": p["nom"],
            "Somme (€)": p["somme"],
            "Taux défaut (%)": p["taux"],
            "Type de fiscalité": p["fiscalite"]["type"],
            "Taux fiscal (%)": p["fiscalite"]["taux"],
            "Nb périodes": len(p.get("periodes", [])),
        }
        for p in st.session_state.placements
    ])
    new_df = st.data_editor(df_inputs, num_rows="dynamic", use_container_width=True, hide_index=True, key="data_editor_inputs")
    for i, (_, row) in enumerate(new_df.iterrows()):
        st.session_state.placements[i]["nom"] = row["Nom"]
        st.session_state.placements[i]["somme"] = row["Somme (€)"]
        st.session_state.placements[i]["taux"] = row["Taux défaut (%)"]
        st.session_state.placements[i]["fiscalite"]["type"] = row["Type de fiscalité"]
        st.session_state.placements[i]["fiscalite"]["taux"] = row["Taux fiscal (%)"]

    idx_pl = st.selectbox("Sélectionner un placement pour éditer les périodes de taux",
                          options=range(len(st.session_state.placements)),
                          format_func=lambda i: st.session_state.placements[i]["nom"])
    pl = st.session_state.placements[idx_pl]
    per_df = pd.DataFrame(pl.get("periodes", []))
    new_per_df = st.data_editor(per_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="periodes_editor")
    st.session_state.placements[idx_pl]["periodes"] = new_per_df.to_dict(orient="records")

st.markdown("## Simuler un changement de conditions entre livrets")
if len(st.session_state.placements) >= 2:
    col_sim1, col_sim2 = st.columns(2)
    with col_sim1:
        idx_source = st.selectbox("Livret à modifier (capital/noms conservés)",
                                 options=range(len(st.session_state.placements)),
                                 format_func=lambda i: st.session_state.placements[i]["nom"])
    with col_sim2:
        idx_modele = st.selectbox("Prendre les conditions de ce livret",
                                 options=[i for i in range(len(st.session_state.placements)) if i != idx_source],
                                 format_func=lambda i: st.session_state.placements[i]["nom"])
    do_swap = st.button("Appliquer les conditions du livret modèle au livret sélectionné")
    if do_swap:
        source, modele = st.session_state.placements[idx_source], st.session_state.placements[idx_modele]
        source["taux"] = modele["taux"]
        source["fiscalite"] = modele["fiscalite"].copy()
        source["periodes"] = [dict(per) for per in modele.get("periodes", [])]
        st.success(f"Conditions de « {modele['nom']} » appliquées à « {source['nom']} ». Comparez l'impact dans les résultats.")

# =========================
# Liste des placements et calculs avec synthèse enrichie (intérêt net annuel et taux distincts listés)
# =========================
st.subheader("Placements (synthèse enrichie)")
if st.session_state.placements:
    debut = st.session_state.periode_globale['debut']
    fin = st.session_state.periode_globale['fin']
    rows = []
    for p in st.session_state.placements:
        df = build_monthly_schedule(p, debut, fin)
        net_annuel = df['Int_net'].sum() if not df.empty else 0.0
        if p.get("periodes"):
            taux_str = ", ".join([f"{per['taux']}% ({per['debut']}→{per['fin']})" for per in p["periodes"]])
        else:
            taux_str = f"{p['taux']}%"
        rows.append({
            "Nom": p["nom"],
            "Somme (€)": p["somme"],
            "Taux défaut/périodes": taux_str,
            "Type fiscalité": p["fiscalite"]["type"],
            "Taux fiscal (%)": p["fiscalite"]["taux"],
            "Intérêt net annuel (€)": round(net_annuel, 2)
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ===== Le reste de votre code pour calculs mensuels, graphiques, exports, etc. ====  
# (Vous pouvez réintégrer ici les sections Graphiques, Résultats mensuels enrichis, Totaux, etc.)
# Conservez votre code stable comme précédemment !

