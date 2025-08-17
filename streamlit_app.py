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
st.caption("Ajoutez ou éditez des placements, importez/exportez des CSV, et visualisez vos résultats.")

# =========================
# Utilitaires Export/Import des entrées
# =========================
CSV_HEADER = [
    "type_ligne", "nom", "somme", "taux_defaut", "fisc_type", "fisc_taux",
    "periode_debut", "periode_fin", "periode_taux"
]

def export_inputs_to_csv(placements: list, periode_globale: dict) -> bytes:
    # ... [INCHANGÉ] ...
    rows = []
    rows.append({
        "type_ligne": "META", "nom": "", "somme": "", "taux_defaut": "", "fisc_type": "",
        "fisc_taux": "", "periode_debut": periode_globale['debut'].strftime("%Y-%m-%d"),
        "periode_fin": periode_globale['fin'].strftime("%Y-%m-%d"), "periode_taux": ""
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
                "type_ligne": "PLACEMENT", "nom": nom, "somme": somme, "taux_defaut": taux_defaut,
                "fisc_type": fisc_type, "fisc_taux": fisc_taux,
                "periode_debut": "", "periode_fin": "", "periode_taux": ""
            })
        else:
            for per in periods:
                rows.append({
                    "type_ligne": "PLACEMENT", "nom": nom, "somme": somme, "taux_defaut": taux_defaut,
                    "fisc_type": fisc_type, "fisc_taux": fisc_taux,
                    "periode_debut": per.get('debut', ''), "periode_fin": per.get('fin', ''), "periode_taux": per.get('taux', '')
                })
    df = pd.DataFrame(rows, columns=CSV_HEADER)
    return df.to_csv(index=False).encode("utf-8")

def import_inputs_from_csv(file_bytes: bytes):
    # ... [INCHANGÉ] ...
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
# [INCHANGÉES — COLLER TELLES QUELLES]
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
    # ... [INCHANGÉE] ...
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
# État session
# =========================
if 'placements' not in st.session_state:
    st.session_state.placements = []

if 'periode_globale' not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {'debut': date(today.year, 1, 1), 'fin': date(today.year, 12, 31)}

if 'edited_dict_list' not in st.session_state:
    st.session_state.edited_dict_list = None   # Pour le Data Editor

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
# Saisie des placements (Ajout uniquement)
# =========================
st.subheader("Ajouter un placement (création)")
with st.expander("Ajouter un placement"):
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        nom = st.text_input("Nom du placement", placeholder="Livret A", key="add_nom")
    with col2:
        somme = st.number_input("Somme investie (€)", min_value=0.0, step=100.0, format="%.2f", key="add_somme")
    with col3:
        taux_defaut = st.number_input("Taux annuel (%) - défaut", min_value=0.0, step=0.05, format="%.3f",
                                      help="Utilisé si aucune période de taux n’est ajoutée.", key="add_taux")

    st.markdown("Fiscalité")
    colf1, colf2 = st.columns([1,1])
    with colf1:
        fiscalite_type = st.selectbox("Type de fiscalité", ["PFU (30%)", "Personnalisé"], index=0, key='add_fisc_type')
    with colf2:
        fiscalite_taux = st.number_input("Taux fiscal (%)", min_value=0.0, max_value=100.0,
                                         value=30.0 if fiscalite_type.startswith("PFU") else 0.0,
                                         step=0.5, format="%.2f", key='add_fisc_taux')

    st.markdown("Périodes de taux (facultatif)")
    st.caption("Ajoutez une ou plusieurs périodes avec des taux différents dans l’année.")
    if 'periodes_temp' not in st.session_state:
        st.session_state.periodes_temp = []

    cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
    with cp1:
        p_deb = st.date_input("Début", value=st.session_state.periode_globale['debut'], key="add_p_deb")
    with cp2:
        p_fin = st.date_input("Fin", value=st.session_state.periode_globale['fin'], key="add_p_fin")
    with cp3:
        p_taux = st.number_input("Taux (%)", min_value=0.0, step=0.05, format="%.3f", key="add_p_taux")
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

# =========================
# MODIFIER ou SUPPRIMER un placement déjà importé ou existant
# =========================
st.divider()
st.subheader("Modifier ou supprimer des placements existants / importés")

def edit_placement_dict_list(placements):
    dict_list = []
    for idx, p in enumerate(placements):
        dict_list.append({
            "index": idx,
            "Nom": p['nom'],
            "Somme (€)": p['somme'],
            "Taux défaut (%)": p['taux'],
            "Fiscalité (%)": p['fiscalite']['taux'],
            "Type Fiscalité": p['fiscalite']['type'],
            "Nb périodes": len(p.get('periodes', [])),
        })
    return dict_list

if st.session_state.placements:
    df_edit = pd.DataFrame(edit_placement_dict_list(st.session_state.placements))
    edited_df = st.data_editor(
        df_edit,
        use_container_width=True,
        num_rows="dynamic",  # Peut ajouter une ligne vide pour insertion
        key="data_editor_placements",
        column_config={
            "index": st.column_config.Column("Index", disabled=True),
            "Nom": st.column_config.TextColumn("Nom", required=True),
            "Type Fiscalité": st.column_config.SelectboxColumn("Type Fiscalité", options=["PFU", "PERSONNALISE"], required=True),
        },
        hide_index=True
    )
    # Appliquer les modifs
    if st.button("Valider les modifications"):
        new_placements = []
        for _, row in edited_df.iterrows():
            if not row["Nom"]:
                continue
            idx = int(row["index"])
            old = st.session_state.placements[idx]
            new_placements.append({
                "nom": row["Nom"],
                "somme": row["Somme (€)"],
                "taux": row["Taux défaut (%)"],
                "fiscalite": {
                    "type": row["Type Fiscalité"],
                    "taux": row["Fiscalité (%)"],
                },
                # On conserve aussi les périodes :
                "periodes": old["periodes"] if "periodes" in old else []
            })
        st.session_state.placements = new_placements
        st.success("Modifications appliquées.")

    # Edition des périodes à part
    exp_per_mod = st.expander("Modifier les périodes d’un placement existant", expanded=False)
    with exp_per_mod:
        choix_nom = st.selectbox("Choisir le placement à modifier", options=[p['nom'] for p in st.session_state.placements])
        for p in st.session_state.placements:
            if p['nom'] == choix_nom:
                periodes = p.get("periodes", [])
                if not periodes:
                    st.info("Aucune période personnalisée, le taux défaut sera utilisé.")
                else:
                    st.dataframe(pd.DataFrame(periodes), use_container_width=True, hide_index=True)
                if st.button(f"Vider les périodes pour « {choix_nom} »"):
                    p["periodes"] = []
                    st.success(f"Périodes vidées pour {choix_nom}.")
                st.caption("Remplacer/ajouter des périodes spécifiques pour ce placement :")
                new_periods = st.text_area(
                    "Entrer les périodes sous la forme : YYYY-MM-DD,YYYY-MM-DD,taux (une par ligne)",
                    value="\n".join([f"{per['debut']},{per['fin']},{per['taux']}" for per in periodes])
                )
                if st.button(f"Appliquer les périodes modifiées à « {choix_nom} »"):
                    new_p = []
                    for line in new_periods.splitlines():
                        if not line.strip():
                            continue
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) == 3:
                            try:
                                new_p.append({
                                    "debut": parts[0],
                                    "fin": parts,
                                    "taux": float(parts),
                                })
                            except Exception:
                                pass
                    p["periodes"] = new_p
                    st.success("Périodes modifiées/supprimées.")

    # SUPPRESSION
    exp_del = st.expander("Supprimer un placement")
    with exp_del:
        del_names = [p['nom'] for p in st.session_state.placements]
        del_selection = st.selectbox("Sélectionner le placement à supprimer", options=del_names, key="del_select")
        if st.button("Supprimer ce placement"):
            st.session_state.placements = [p for p in st.session_state.placements if p['nom'] != del_selection]
            st.success(f"Placement « {del_selection} » supprimé.")

else:
    st.info("Aucun placement à modifier pour l’instant.")

# =========================
# [Reste du code d’analyse, graphiques, export — INCHANGÉ]
# =========================
# [COLLER LA PARTIE ANALYSE, GRAPHIQUES, EXPORTS,
# TOUT CE QUI SUIT « Liste des placements et calculs » DANS TON CODE INITIAL]
