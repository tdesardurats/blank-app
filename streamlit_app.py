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
st.caption("Ajoutez des placements, définissez les périodes de taux, la fiscalité, éditez après import, et visualisez brut/net. Export/Import CSV inclus.")

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
    # Ligne META
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

    # Lignes PLACEMENT
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
# Saisie des placements (ajout)
# =========================
st.subheader("Ajouter un placement")
with st.expander("Ajouter un placement"):
    col1, col2, col3 = st.columns([2,1,1])
    with col1:
        nom_add = st.text_input("Nom du placement", placeholder="Livret A", key="add_nom")
    with col2:
        somme_add = st.number_input("Somme investie (€)", min_value=0.0, step=100.0, format="%.2f", key="add_somme")
    with col3:
        taux_defaut_add = st.number_input("Taux annuel (%) - défaut", min_value=0.0, step=0.05, format="%.3f", key="add_taux")

    st.markdown("Fiscalité")
    colf1, colf2 = st.columns([1,1])
    with colf1:
        fiscalite_type_add = st.selectbox("Type de fiscalité", ["PFU (30%)", "Personnalisé"], index=0, key='add_fisc_type')
    with colf2:
        fiscalite_taux_add = st.number_input("Taux fiscal (%)", min_value=0.0, max_value=100.0,
                                             value=30.0 if fiscalite_type_add.startswith("PFU") else 0.0,
                                             step=0.5, format="%.2f", key='add_fisc_taux')

    st.markdown("Périodes de taux (facultatif)")
    if 'periodes_temp' not in st.session_state:
        st.session_state.periodes_temp = []

    cp1, cp2, cp3, cp4 = st.columns([1,1,1,1])
    with cp1:
        p_deb_add = st.date_input("Début", value=st.session_state.periode_globale['debut'], key="add_p_deb")
    with cp2:
        p_fin_add = st.date_input("Fin", value=st.session_state.periode_globale['fin'], key="add_p_fin")
    with cp3:
        p_taux_add = st.number_input("Taux (%)", min_value=0.0, step=0.05, format="%.3f", key="add_p_taux")
    with cp4:
        if st.button("Ajouter période", key="btn_add_period"):
            if p_deb_add <= p_fin_add:
                st.session_state.periodes_temp.append({
                    'debut': p_deb_add.strftime("%Y-%m-%d"),
                    'fin': p_fin_add.strftime("%Y-%m-%d"),
                    'taux': float(p_taux_add)
                })
            else:
                st.warning("La date de début de période doit précéder la date de fin.")

    if st.session_state.periodes_temp:
        st.dataframe(pd.DataFrame(st.session_state.periodes_temp), use_container_width=True, hide_index=True)
        if st.button("Vider les périodes", key="btn_clear_periods_add"):
            st.session_state.periodes_temp = []

    if st.button("Enregistrer le placement", key="btn_save_placement"):
        if not nom_add:
            st.warning("Veuillez saisir un nom de placement.")
        else:
            placement = {
                'uid': str(uuid.uuid4()),
                'nom': nom_add.strip(),
                'somme': float(somme_add),
                'taux': float(taux_defaut_add),
                'fiscalite': {
                    'type': 'PFU' if fiscalite_type_add.startswith("PFU") else 'PERSONNALISE',
                    'taux': float(fiscalite_taux_add)
                },
                'periodes': st.session_state.periodes_temp.copy()
            }
            st.session_state.placements.append(placement)
            st.session_state.periodes_temp = []
            st.success(f"Placement « {placement['nom']} » enregistré.")

# =========================
# Section d'édition fiable (placements)
# =========================
st.divider()
st.subheader("Modifier/Supprimer les placements existants")

def placements_to_editor_df(pls: list) -> pd.DataFrame:
    rows = []
    for p in pls:
        if "uid" not in p or not p["uid"]:
            p["uid"] = str(uuid.uuid4())
        rows.append({
            "uid": p["uid"],
            "Nom": p['nom'],
            "Somme (€)": float(p['somme']),
            "Taux défaut (%)": float(p['taux']),
            "Type Fiscalité": p['fiscalite']['type'],
            "Fiscalité (%)": float(p['fiscalite']['taux']),
            "Nb périodes": len(p.get('periodes', []))
        })
    df = pd.DataFrame(rows)
    return df

def normalize_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def apply_editor_changes():
    edits = st.session_state.get("edits", {})
    base_df: pd.DataFrame = st.session_state.get("editor_df", pd.DataFrame()).copy()
    if base_df.empty:
        return

    # Appliquer cellules modifiées
    for row_idx, change in edits.get("edited_rows", {}).items():
        for col, val in change.items():
            base_df.at[row_idx, col] = val

    # Lignes ajoutées
    for added in edits.get("added_rows", []):
        new_row = {c: added.get(c, None) for c in base_df.columns}
        if not new_row.get("uid"):
            new_row["uid"] = str(uuid.uuid4())
        base_df.loc[len(base_df)] = new_row

    # Lignes supprimées
    deleted = edits.get("deleted_rows", [])
    if deleted:
        base_df = base_df.drop(index=deleted).reset_index(drop=True)

    # Sauver le DF modifié
    st.session_state["editor_df"] = base_df

    # Reconstruire placements via uid
    uid_to_periods = {p["uid"]: p.get("periodes", []) for p in st.session_state.placements}
    new_placements = []
    for _, r in base_df.iterrows():
        uid = str(r.get("uid"))
        nom = str(r.get("Nom") or "").strip()
        if not nom:
            continue
        somme = normalize_float(r.get("Somme (€)"), 0.0)
        taux_def = normalize_float(r.get("Taux défaut (%)"), 0.0)
        fisc_type = str(r.get("Type Fiscalité") or "PFU")
        if fisc_type not in ("PFU", "PERSONNALISE"):
            fisc_type = "PFU"
        fisc_taux = normalize_float(r.get("Fiscalité (%)"), 30.0)
        new_placements.append({
            "uid": uid,
            "nom": nom,
            "somme": somme,
            "taux": taux_def,
            "fiscalite": {"type": fisc_type, "taux": fisc_taux},
            "periodes": uid_to_periods.get(uid, [])
        })

    st.session_state.placements = new_placements
    st.toast("Modifications appliquées.", icon="✅")

# Construire DF pour éditeur
editor_df = placements_to_editor_df(st.session_state.placements)
st.session_state["editor_df"] = editor_df.copy()

st.data_editor(
    editor_df,
    key="edits",
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "uid": st.column_config.Column("uid", disabled=True),
        "Nom": st.column_config.TextColumn("Nom", required=True),
        "Type Fiscalité": st.column_config.SelectboxColumn("Type Fiscalité", options=["PFU", "PERSONNALISE"], required=True),
        "Somme (€)": st.column_config.NumberColumn("Somme (€)", step=50.0, format="%.2f"),
        "Taux défaut (%)": st.column_config.NumberColumn("Taux défaut (%)", step=0.05, format="%.3f"),
        "Fiscalité (%)": st.column_config.NumberColumn("Fiscalité (%)", step=0.5, format="%.2f"),
        "Nb périodes": st.column_config.Column("Nb périodes", disabled=True),
    },
    on_change=apply_editor_changes,
)

# =========================
# Édition des périodes par placement (sélection stable)
# =========================
st.markdown("### Éditer les périodes d’un placement")

def ensure_uid_on_placements():
    changed = False
    for p in st.session_state.placements:
        if "uid" not in p or not p["uid"]:
            p["uid"] = str(uuid.uuid4())
            changed = True
    return changed

if st.session_state.placements:
    if ensure_uid_on_placements():
        st.rerun()

    uid_options = [p["uid"] for p in st.session_state.placements]
    uid_to_label = {p["uid"]: f"{p['nom']} — {p['uid'][:8]}" for p in st.session_state.placements}

    if "period_uid_selected" not in st.session_state or st.session_state.get("period_uid_selected") not in uid_options:
        st.session_state.period_uid_selected = uid_options[0] if uid_options else None

    selected_uid = st.selectbox(
        "Choisir le placement",
        options=uid_options,
        index=uid_options.index(st.session_state.period_uid_selected) if st.session_state.period_uid_selected in uid_options else 0,
        format_func=lambda u: uid_to_label.get(u, u),
        key="period_select_uid_controlled"
    )

    if selected_uid != st.session_state.period_uid_selected:
        st.session_state.period_uid_selected = selected_uid

    chosen = next((p for p in st.session_state.placements if p["uid"] == st.session_state.period_uid_selected), None)

    if chosen:
        st.write(f"Périodes actuelles pour « {chosen['nom']} »")
        periods_df = pd.DataFrame(chosen.get("periodes", []))
        if periods_df.empty:
            periods_df = pd.DataFrame(columns=["debut","fin","taux"])
        st.dataframe(periods_df, use_container_width=True, hide_index=True)

        form_key = f"edit_periods_form_{chosen['uid']}"
        with st.form(form_key):
            c1, c2, c3 = st.columns(3)
            with c1:
                new_deb = st.date_input("Début", value=st.session_state.periode_globale['debut'], key=f"deb_{chosen['uid']}")
            with c2:
                new_fin = st.date_input("Fin", value=st.session_state.periode_globale['fin'], key=f"fin_{chosen['uid']}")
            with c3:
                new_taux = st.number_input("Taux (%)", min_value=0.0, step=0.05, format="%.3f", key=f"taux_{chosen['uid']}")

            submitted_add = st.form_submit_button("Ajouter cette période")
            if submitted_add:
                if new_deb <= new_fin:
                    chosen.setdefault("periodes", []).append({
                        "debut": new_deb.strftime("%Y-%m-%d"),
                        "fin": new_fin.strftime("%Y-%m-%d"),
                        "taux": float(new_taux)
                    })
                    st.success("Période ajoutée.")
                    st.rerun()
                else:
                    st.error("Début doit précéder fin.")

        if chosen.get("periodes"):
            del_idx = st.number_input(
                "Supprimer la période n° (index commençant à 0)",
                min_value=0, max_value=len(chosen["periodes"])-1, step=1, value=0,
                key=f"del_idx_{chosen['uid']}"
            )
            if st.button("Supprimer cette période", key=f"btn_del_period_{chosen['uid']}"):
                try:
                    chosen["periodes"].pop(int(del_idx))
                    st.success("Période supprimée.")
                    st.rerun()
                except Exception:
                    st.error("Index invalide.")

        if st.button("Vider toutes les périodes de ce placement", key=f"btn_clear_periods_{chosen['uid']}"):
            chosen["periodes"] = []
            st.success("Toutes les périodes ont été supprimées.")
            st.rerun()
else:
    st.info("Aucun placement pour éditer des périodes.")

# =========================
# Liste des placements et calculs
# =========================
st.subheader("Placements")
if not st.session_state.placements:
    st.info("Aucun placement pour l’instant. Ajoutez-en via le panneau ci-dessus.")
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
    all_monthly = []
    for p in st.session_state.placements:
        dfp = build_monthly_schedule(p, debut, fin, base_jour=base_jour)
        all_monthly.append(dfp)
    monthly = pd.concat(all_monthly, ignore_index=True) if all_monthly else pd.DataFrame(
        columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour']
    )

    # Correctif: s'assurer que Date est bien datetime avant toute utilisation de .dt
    if not monthly.empty:
        monthly['Date'] = pd.to_datetime(monthly['Date'], errors='coerce')

    # Tableau mensuel enrichi
    st.markdown("### Résultats mensuels enrichis (brut/net + nb_jours + moyennes/jour)")
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

    # Totaux par placement
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

    # =========================
    # Graphiques Plotly
    # =========================
    st.markdown("## Graphiques d’évolution (Plotly)")

    monthly_sorted = monthly.sort_values('Date').copy()
    if not monthly_sorted.empty:
        monthly_sorted['Date'] = pd.to_datetime(monthly_sorted['Date'], errors='coerce')
        placements_list = monthly_sorted['Placement'].unique().tolist()
        color_palette = px.colors.qualitative.Safe
        color_map = {plc: color_palette[i % len(color_palette)] for i, plc in enumerate(placements_list)}

        # Par placement - Intérêt mensuel (Brut vs Net)
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

        # Par placement - Cumul des intérêts (Net)
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

        # Cumul annuel par placement (couleur par placement)
        st.markdown("### Cumul annuel par placement (couleur par placement)")
        cum_by_pl = monthly_sorted.copy().sort_values(['Placement','Date'])
        cum_by_pl['Net_cumulé'] = cum_by_pl.groupby('Placement')['Int_net'].cumsum()

        fig_cumul = go.Figure()
        for nom_pl in placements_list:
            sub = cum_by_pl[cum_by_pl['Placement'] == nom_pl]
            fig_cumul.add_trace(go.Scatter(
                x=sub['Date'], y=sub['Net_cumulé'],
                mode='lines+markers',
                name=nom_pl,
                line=dict(color=color_map[nom_pl], width=2),
                marker=dict(size=6),
                hovertemplate="Mois: %{x|%Y-%m}<br>Net cumulé: %{y:.2f} €<extra></extra>"
            ))
        fig_cumul.update_layout(
            title_text="Évolution annuelle du net cumulé par placement",
            xaxis_title="Mois",
            yaxis_title="€",
            legend_title_text="Placement",
            margin=dict(t=50, l=40, r=20, b=40),
            height=420,
            template="plotly_white"
        )
        st.plotly_chart(fig_cumul, use_container_width=True)

        # Cumul global empilé (net)
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
    # Export des résultats
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
            for p in placements_imp:
                p.setdefault("uid", str(uuid.uuid4()))
            st.session_state.placements = placements_imp
            st.session_state.periode_globale = periode_imp
            st.success("Données d’entrée importées avec succès. Éditez-les ci-dessus si besoin.")
        except Exception as e:
            st.error(f"Erreur lors de l’import: {e}")

# =========================
# Notes
# =========================
st.divider()
with st.expander("Notes & limites"):
    st.markdown(
        "- Intérêts calculés en prorata linéaire sur base jours/an configurable (par défaut 365).\n"
        "- La fiscalité est appliquée comme un pourcentage unique sur les intérêts (modèle simple). Pour des cas réels (PFU 12.8% + PS 17.2%, exonérations), adapter au besoin.\n"
        "- Les périodes de taux modélisent des changements en cours d'année; en l’absence de périodes, le taux défaut s’applique.\n"
        "- Tableau mensuel enrichi avec nb_jours (taille du mois) et moyennes/jour brut & net.\n"
        "- Graphiques Plotly: barres mensuelles, cumuls par placement, cumul global empilé.\n"
        "- Édition fiable via st.data_editor + on_change + uid pour une persistance immédiate et stable."
    )

