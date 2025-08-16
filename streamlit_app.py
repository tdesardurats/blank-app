import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import io

# --- Config et variables d'état ---
st.set_page_config(page_title="Livrets : Paramétrage & Simulation", layout="wide", page_icon="💶")
st.title("💶 Simulateur livrets : paramètres, fiscalité, périodes, analyse")
st.caption("Ajoutez des placements ou importez-les, modifiez tout à volonté, comparez l’impact d’un changement de taux et exportez vos résultats.")

CSV_HEADER = [
    "type_ligne", "nom", "somme", "taux_defaut", "fisc_type", "fisc_taux",
    "periode_debut", "periode_fin", "periode_taux"
]

if 'placements' not in st.session_state:
    st.session_state.placements = []
if 'periode_globale' not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {'debut': date(today.year, 1, 1), 'fin': date(today.year, 12, 31)}

# --- Barre latérale : paramètres globaux (toujours affichée) ---
with st.sidebar:
    st.header("Période d'analyse et base de calcul")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        debut = st.date_input("Début période", value=st.session_state.periode_globale['debut'])
    with col_g2:
        fin = st.date_input("Fin période", value=st.session_state.periode_globale['fin'])
    if debut > fin:
        st.error("La date de début doit être avant la date de fin.")
    st.session_state.periode_globale = {'debut': debut, 'fin': fin}
    base_jour = st.number_input("Base jours/an", min_value=360, max_value=366, value=365, step=1, help="365 par défaut.")

# --- Ajout manuel d'un placement (toujours visible) ---
st.markdown("## Ajouter un placement (manuel)")

with st.form("ajout_placement_form", clear_on_submit=True):
    col1, col2, col3 = st.columns([2,1,1])
    with col1:  nom = st.text_input("Nom du placement", placeholder="Livret A")
    with col2:  somme = st.number_input("Somme investie (€)", min_value=0.0, step=100.0, format="%.2f")
    with col3:  taux_defaut = st.number_input("Taux annuel (%) - défaut", min_value=0.0, step=0.05, format="%.3f")
    fiscalite_type = st.selectbox("Type de fiscalité", ["PFU (30%)", "Personnalisé"], index=0)
    fiscalite_taux = st.number_input("Taux fiscal (%)", min_value=0.0, max_value=100.0,
                                    value=30.0 if fiscalite_type.startswith("PFU") else 0.0, step=0.5, format="%.2f")
    submit_manual = st.form_submit_button("Ajouter ce placement")
    if submit_manual and nom:
        st.session_state.placements.append({
            'nom': nom.strip(),
            'somme': somme,
            'taux': taux_defaut,
            'fiscalite': {
                'type': 'PFU' if fiscalite_type.startswith("PFU") else 'PERSONNALISE',
                'taux': fiscalite_taux
            },
            'periodes': []
        })
        st.success(f"Placement « {nom} » ajouté.")

# --- Import / Export des données d’entrée (toujours visible) ---
st.markdown("## Import / export du paramétrage")
col_exp, col_imp = st.columns(2)
def export_inputs_to_csv(placements, periode_globale):
    rows = []
    rows.append({
        "type_ligne": "META",
        "nom": "", "somme": "", "taux_defaut": "", "fisc_type": "", "fisc_taux": "",
        "periode_debut": periode_globale['debut'].strftime("%Y-%m-%d"),
        "periode_fin": periode_globale['fin'].strftime("%Y-%m-%d"),
        "periode_taux": ""
    })
    for p in placements:
        periods = p.get('periodes', [])
        if not periods:
            rows.append({
                "type_ligne": "PLACEMENT", "nom": p['nom'], "somme": p['somme'], "taux_defaut": p['taux'],
                "fisc_type": p['fiscalite']['type'], "fisc_taux": p['fiscalite']['taux'],
                "periode_debut": "", "periode_fin": "", "periode_taux": ""
            })
        else:
            for per in periods:
                rows.append({
                    "type_ligne": "PLACEMENT", "nom": p['nom'], "somme": p['somme'], "taux_defaut": p['taux'],
                    "fisc_type": p['fiscalite']['type'], "fisc_taux": p['fiscalite']['taux'],
                    "periode_debut": per.get('debut',''), "periode_fin":per.get('fin',''), "periode_taux": per.get('taux','')
                })
    df = pd.DataFrame(rows, columns=CSV_HEADER)
    return df.to_csv(index=False).encode("utf-8")

def import_inputs_from_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
    meta = df[df['type_ligne'] == 'META']
    if not meta.empty:
        m0 = meta.iloc[0]
        try:
            g_debut = datetime.strptime(m0['periode_debut'], "%Y-%m-%d").date()
            g_fin = datetime.strptime(m0['periode_fin'], "%Y-%m-%d").date()
        except:  # fallback
            today = date.today()
            g_debut, g_fin = date(today.year,1,1), date(today.year,12,31)
    else:
        today = date.today()
        g_debut, g_fin = date(today.year,1,1), date(today.year,12,31)
    periode_globale = {"debut": g_debut, "fin": g_fin}
    plc_rows = df[df['type_ligne'] == 'PLACEMENT'].copy()
    to_float_safe = lambda x, d=0.0: float(str(x).replace(',','.')) if str(x).replace('.', '', 1).isdigit() else d
    placements_dict = {}
    for _, r in plc_rows.iterrows():
        nom = r['nom'].strip()
        if not nom: continue
        if nom not in placements_dict:
            placements_dict[nom] = {
                "nom": nom, "somme": to_float_safe(r['somme'], 0.0), "taux": to_float_safe(r['taux_defaut'], 0.0),
                "fiscalite": {"type": r['fisc_type'], "taux": to_float_safe(r['fisc_taux'], 30.0)},
                "periodes": []
            }
        per_deb, per_fin, per_taux = r['periode_debut'].strip(), r['periode_fin'].strip(), r['periode_taux'].strip()
        if per_deb and per_fin and per_taux:
            placements_dict[nom]["periodes"].append({"debut": per_deb, "fin": per_fin, "taux": to_float_safe(per_taux)})
    placements = list(placements_dict.values())
    return placements, periode_globale

with col_exp:
    if st.button("📤 Exporter les données d’entrée (CSV)", use_container_width=True):
        csv_in_bytes = export_inputs_to_csv(st.session_state.placements, st.session_state.periode_globale)
        st.download_button("Télécharger le CSV des données d’entrée", data=csv_in_bytes, file_name="donnees_entree_livrets.csv", mime="text/csv", use_container_width=True)
with col_imp:
    uploaded = st.file_uploader("📥 Importer un CSV de données d’entrée", type=["csv"])
    if uploaded is not None:
        try:
            placements_imp, periode_imp = import_inputs_from_csv(uploaded.read())
            st.session_state.placements = placements_imp
            st.session_state.periode_globale = periode_imp
            st.success("Données d’entrée importées avec succès.")
        except Exception as e:
            st.error(f"Erreur lors de l’import: {e}")

# --- Edition (toujours visible si placements) ---
if st.session_state.placements:
    st.markdown("## Modification et gestion avancée des placements")
    # Edition interactive des placements (hors périodes)
    edit_df = pd.DataFrame([
        {
            "Nom": p["nom"],
            "Somme (€)": p["somme"],
            "Taux défaut (%)": p["taux"],
            "Type de fiscalité": p["fiscalite"]["type"],
            "Taux fiscal (%)": p["fiscalite"]["taux"],
            "Nb périodes": len(p.get("periodes", [])),
        } for p in st.session_state.placements
    ])
    edited = st.data_editor(edit_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="data_editor_inputs")
    for i, (_, row) in enumerate(edited.iterrows()):
        st.session_state.placements[i]["nom"] = row["Nom"]
        st.session_state.placements[i]["somme"] = row["Somme (€)"]
        st.session_state.placements[i]["taux"] = row["Taux défaut (%)"]
        st.session_state.placements[i]["fiscalite"]["type"] = row["Type de fiscalité"]
        st.session_state.placements[i]["fiscalite"]["taux"] = row["Taux fiscal (%)"]

    idx_pl = st.selectbox("Éditer les périodes de taux pour :", options=range(len(st.session_state.placements)), format_func=lambda i: st.session_state.placements[i]["nom"])
    pl = st.session_state.placements[idx_pl]
    per_df = pd.DataFrame(pl.get("periodes", []))
    new_per_df = st.data_editor(per_df, num_rows="dynamic", use_container_width=True, hide_index=True, key="periodes_editor")
    st.session_state.placements[idx_pl]["periodes"] = new_per_df.to_dict(orient="records")

# Simulateur bascule de conditions — toujours visible si ≥2 placements
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

# Tableau synthèse des placements avancé
st.subheader("Placements (synthèse enrichie)")
if st.session_state.placements:
    debut = st.session_state.periode_globale['debut']
    fin = st.session_state.periode_globale['fin']
    rows = []
    for p in st.session_state.placements:
        df = build_monthly_schedule(p, debut, fin, base_jour=base_jour)
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

# (Tu peux ajouter ici tous tes blocs graphiques, exports, résultats détaillés)
