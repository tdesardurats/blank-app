import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date, datetime
from dateutil.relativedelta import relativedelta
import io
import uuid

# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Livrets: intérêts, fiscalité, périodes & graphes", page_icon="💶", layout="wide")
st.title("💶 Visualiseur d'intérêts avec fiscalité, périodes de taux, tableaux et graphes")
st.caption("Éditez directement dans les tableaux. Les données éditées sont conservées en session tant que vous ne réimportez pas de CSV.")

# =========================================================
# Helpers
# =========================================================
CSV_HEADER = [
    "type_ligne","nom","somme","taux_defaut","fisc_type","fisc_taux",
    "periode_debut","periode_fin","periode_taux"
]

def to_float_safe(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except Exception:
        return default

def valid_date(s):
    try:
        datetime.strptime(str(s), "%Y-%m-%d")
        return True
    except Exception:
        return False

def export_inputs_to_csv(placements: list, periode_globale: dict) -> bytes:
    rows = []
    rows.append({
        "type_ligne": "META","nom": "","somme": "","taux_defaut": "","fisc_type": "","fisc_taux": "",
        "periode_debut": periode_globale['debut'].strftime("%Y-%m-%d"),
        "periode_fin": periode_globale['fin'].strftime("%Y-%m-%d"),
        "periode_taux": ""
    })
    for p in placements:
        nom = p.get('nom',''); somme = p.get('somme',0.0); taux_defaut = p.get('taux',0.0)
        fisc_type = p.get('fiscalite',{}).get('type','PFU'); fisc_taux = p.get('fiscalite',{}).get('taux',30.0)
        periods = p.get('periodes',[])
        if not periods:
            rows.append({
                "type_ligne":"PLACEMENT","nom":nom,"somme":somme,"taux_defaut":taux_defaut,
                "fisc_type":fisc_type,"fisc_taux":fisc_taux,
                "periode_debut":"","periode_fin":"","periode_taux":""
            })
        else:
            for per in periods:
                rows.append({
                    "type_ligne":"PLACEMENT","nom":nom,"somme":somme,"taux_defaut":taux_defaut,
                    "fisc_type":fisc_type,"fisc_taux":fisc_taux,
                    "periode_debut": per.get('debut',''),
                    "periode_fin": per.get('fin',''),
                    "periode_taux": per.get('taux','')
                })
    df = pd.DataFrame(rows, columns=CSV_HEADER)
    return df.to_csv(index=False).encode("utf-8")

def import_inputs_from_csv(file_bytes: bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
    missing = [c for c in CSV_HEADER if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")

    # Période globale
    meta = df[df['type_ligne']=="META"]
    if not meta.empty:
        m0 = meta.iloc[0]
        try:
            g_debut = datetime.strptime(m0['periode_debut'],"%Y-%m-%d").date()
            g_fin   = datetime.strptime(m0['periode_fin'],"%Y-%m-%d").date()
        except Exception:
            today = date.today()
            g_debut = date(today.year,1,1); g_fin = date(today.year,12,31)
    else:
        today = date.today()
        g_debut = date(today.year,1,1); g_fin = date(today.year,12,31)
    periode_globale = {"debut": g_debut, "fin": g_fin}

    plc_rows = df[df['type_ligne']=="PLACEMENT"].copy()

    # Construit 2 tables: placements et periodes (tabulaires, source unique de vérité pour l'éditeur)
    placements_dict = {}
    periods_list = []
    for _, r in plc_rows.iterrows():
        nom = r['nom'].strip()
        if not nom:
            continue
        if nom not in placements_dict:
            uid = str(uuid.uuid4())
            placements_dict[nom] = {
                "uid": uid,
                "Nom": nom,
                "Somme (€)": to_float_safe(r['somme'],0.0),
                "Taux défaut (%)": to_float_safe(r['taux_defaut'],0.0),
                "Type Fiscalité": r['fisc_type'] if r['fisc_type'] in ("PFU","PERSONNALISE") else "PFU",
                "Fiscalité (%)": to_float_safe(r['fisc_taux'],30.0)
            }
        p_deb = r['periode_debut'].strip(); p_fin = r['periode_fin'].strip(); p_tx = r['periode_taux'].strip()
        if p_deb and p_fin and p_tx:
            periods_list.append({
                "uid_placement": placements_dict[nom]["uid"],
                "Début": p_deb,
                "Fin": p_fin,
                "Taux (%)": to_float_safe(p_tx, placements_dict[nom]["Taux défaut (%)"])
            })

    pl_table = pd.DataFrame(list(placements_dict.values())) if placements_dict else pd.DataFrame(columns=[
        "uid","Nom","Somme (€)","Taux défaut (%)","Type Fiscalité","Fiscalité (%)"
    ])
    per_table = pd.DataFrame(periods_list) if periods_list else pd.DataFrame(columns=["uid_placement","Début","Fin","Taux (%)"])
    return pl_table, per_table, periode_globale

# Calculs
def mois_range(start: date, end: date):
    cur = date(start.year,start.month,1)
    last = date(end.year,end.month,1)
    res=[]
    while cur<=last:
        res.append(cur)
        cur = (cur + relativedelta(months=1))
    return res

def nb_jours_mois(d: date):
    nxt = d + relativedelta(months=1)
    return (nxt - d).days

def clip_period_to_month(period_start: date, period_end: date, month_start: date) -> int:
    month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
    s=max(period_start,month_start); e=min(period_end,month_end)
    if e < s: return 0
    return (e - s).days + 1

def parse_date(s: str) -> date:
    return datetime.strptime(s,"%Y-%m-%d").date()

def compute_brut_net_for_month(capital: float, taux_annuel: float, jours: int, base_jour: int, tax_rate: float):
    interet_brut = capital * (taux_annuel/100.0) * (jours/base_jour)
    interet_net = interet_brut * (1.0 - tax_rate/100.0)
    return interet_brut, interet_net

def build_monthly_schedule_from_tables(pl_row: pd.Series, per_table: pd.DataFrame, start_global: date, end_global: date, base_jour=365):
    nom = pl_row["Nom"]
    capital = float(pl_row["Somme (€)"])
    tax_rate = float(pl_row["Fiscalité (%)"])
    default_rate = float(pl_row["Taux défaut (%)"])
    uid = pl_row["uid"]

    # Récup périodes pour ce placement
    periods_df = per_table[per_table["uid_placement"]==uid].copy()

    # Si aucune période => utiliser le taux défaut pour toute la plage
    periods = []
    if periods_df.empty:
        periods = [{
            'debut': start_global.strftime("%Y-%m-%d"),
            'fin': end_global.strftime("%Y-%m-%d"),
            'taux': default_rate
        }]
    else:
        for _, r in periods_df.iterrows():
            d=r['Début']; f=r['Fin']; t=r['Taux (%)']
            if not valid_date(d) or not valid_date(f):
                continue
            periods.append({'debut': d, 'fin': f, 'taux': float(t)})

    # Normalisation/troncature sur la période globale
    norm_periods=[]
    for p in periods:
        deb = parse_date(p['debut']); fin = parse_date(p['fin'])
        if fin<start_global or deb>end_global: continue
        deb=max(deb,start_global); fin=min(fin,end_global)
        if deb<=fin:
            norm_periods.append({'debut':deb,'fin':fin,'taux':float(p['taux'])})
    if not norm_periods:
        return pd.DataFrame(columns=['Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour'])

    months = mois_range(start_global,end_global)
    rows=[]
    for m_start in months:
        interet_brut_m=0.0; interet_net_m=0.0; jours_pond=0
        taux_expl=[]
        for p in norm_periods:
            j=clip_period_to_month(p['debut'],p['fin'],m_start)
            if j<=0: continue
            ib,in_ = compute_brut_net_for_month(capital,p['taux'],j,base_jour,tax_rate)
            interet_brut_m += ib; interet_net_m += in_
            jours_pond += j
            taux_expl.append((p['taux'], j))
        taux_aff = (sum(t*j for t,j in taux_expl)/sum(j for _,j in taux_expl)) if taux_expl else 0.0
        rows.append({
            'Placement': nom, 'Date': m_start, 'Capital': capital, 'Taux(%)': round(taux_aff,4),
            'Jours_pondérés': int(jours_pond), 'Int_brut': interet_brut_m, 'Int_net': interet_net_m
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['Int_brut']=df['Int_brut'].round(2); df['Int_net']=df['Int_net'].round(2)
        df['nb_jours']=df['Date'].apply(nb_jours_mois)
        df['Brut_moyen_jour']=(df['Int_brut']/df['nb_jours']).round(4)
        df['Net_moyen_jour']=(df['Int_net']/df['nb_jours']).round(4)
        df['Date']=pd.to_datetime(df['Date'])
    return df

def reconstruct_placements_from_tables(pl_table: pd.DataFrame, per_table: pd.DataFrame) -> list:
    placements=[]
    if pl_table is None or pl_table.empty:
        return placements
    for _, r in pl_table.iterrows():
        nom = str(r.get("Nom") or "").strip()
        if not nom:
            continue
        uid = str(r.get("uid") or str(uuid.uuid4()))
        fisc_type = str(r.get("Type Fiscalité") or "PFU")
        if fisc_type not in ("PFU","PERSONNALISE"):
            fisc_type = "PFU"
        p_dict = {
            "uid": uid,
            "nom": nom,
            "somme": to_float_safe(r.get("Somme (€)"), 0.0),
            "taux": to_float_safe(r.get("Taux défaut (%)"), 0.0),
            "fiscalite": {"type": fisc_type, "taux": to_float_safe(r.get("Fiscalité (%)"), 30.0)},
            "periodes": []
        }
        # Ajout des périodes
        if per_table is not None and not per_table.empty:
            sub = per_table[per_table["uid_placement"]==uid]
            for _, pr in sub.iterrows():
                deb=str(pr.get("Début","")); fin=str(pr.get("Fin","")); tx=to_float_safe(pr.get("Taux (%)"), None)
                if deb and fin and tx is not None and valid_date(deb) and valid_date(fin):
                    p_dict["periodes"].append({"debut":deb,"fin":fin,"taux":float(tx)})
        placements.append(p_dict)
    return placements

# =========================================================
# Session state initialization
# =========================================================
if "periode_globale" not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {"debut": date(today.year,1,1), "fin": date(today.year,12,31)}
if "pl_table" not in st.session_state:
    st.session_state.pl_table = pd.DataFrame(columns=["uid","Nom","Somme (€)","Taux défaut (%)","Type Fiscalité","Fiscalité (%)"])
if "per_table" not in st.session_state:
    st.session_state.per_table = pd.DataFrame(columns=["uid_placement","Début","Fin","Taux (%)"])
if "placements" not in st.session_state:
    st.session_state.placements = []

# =========================================================
# Sidebar paramètres globaux
# =========================================================
with st.sidebar:
    st.header("Paramètres globaux")
    col1, col2 = st.columns(2)
    with col1:
        g_deb = st.date_input("Début période", value=st.session_state.periode_globale["debut"], key="glob_deb")
    with col2:
        g_fin = st.date_input("Fin période", value=st.session_state.periode_globale["fin"], key="glob_fin")
    if g_deb > g_fin:
        st.error("La date de début doit être avant la date de fin.")
    st.session_state.periode_globale = {"debut": g_deb, "fin": g_fin}

    st.markdown("---")
    base_jour = st.number_input("Base jours/an", min_value=360, max_value=366, value=365, step=1, help="365 par défaut; 360 possible.")

    st.markdown("---")
    auto_apply = st.checkbox("Modifications automatiques (sans bouton)", value=True, help="Applique immédiatement les changements des tableaux.")

# =========================================================
# Import/Export des données d’entrée
# =========================================================
st.markdown("## Import / Export des données d’entrée")
col_exp, col_imp = st.columns(2)
with col_imp:
    uploaded = st.file_uploader("📥 Importer un CSV de données d’entrée", type=["csv"])
    if uploaded is not None:
        try:
            pl_tab, per_tab, periode_imp = import_inputs_from_csv(uploaded.read())
            # IMPORTANT: on remplace les tables d'édition UNIQUEMENT ici
            st.session_state.pl_table = pl_tab.copy()
            st.session_state.per_table = per_tab.copy()
            st.session_state.periode_globale = periode_imp
            # Reconstruire placements pour initialiser les calculs
            st.session_state.placements = reconstruct_placements_from_tables(st.session_state.pl_table, st.session_state.per_table)
            st.success("CSV importé. Modifiez directement dans les tableaux ci-dessous.")
        except Exception as e:
            st.error(f"Erreur import: {e}")

with col_exp:
    # Exporter à partir du modèle placements (reconstruit depuis les tableaux)
    pl_for_export = reconstruct_placements_from_tables(st.session_state.pl_table, st.session_state.per_table)
    csv_in_bytes = export_inputs_to_csv(pl_for_export, st.session_state.periode_globale)
    st.download_button("📤 Exporter les données d’entrée (CSV)", data=csv_in_bytes, file_name="donnees_entree_livrets.csv", mime="text/csv", use_container_width=True)

st.divider()

# =========================================================
# Tableaux éditables (source unique = session_state tables)
# =========================================================
st.subheader("Placements (édition directe)")
# S'assurer qu'un uid existe pour chaque ligne
if not st.session_state.pl_table.empty:
    if "uid" not in st.session_state.pl_table.columns:
        st.session_state.pl_table["uid"] = [str(uuid.uuid4()) for _ in range(len(st.session_state.pl_table))]
    else:
        # Remplir uid manquants
        st.session_state.pl_table["uid"] = st.session_state.pl_table["uid"].apply(lambda x: x if isinstance(x,str) and x.strip() else str(uuid.uuid4()))

pl_editor = st.data_editor(
    st.session_state.pl_table,
    key="pl_editor",
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "uid": st.column_config.Column("uid", disabled=True),
        "Nom": st.column_config.TextColumn("Nom", required=True),
        "Somme (€)": st.column_config.NumberColumn("Somme (€)", step=50.0, format="%.2f"),
        "Taux défaut (%)": st.column_config.NumberColumn("Taux défaut (%)", step=0.05, format="%.3f"),
        "Type Fiscalité": st.column_config.SelectboxColumn("Type Fiscalité", options=["PFU","PERSONNALISE"], required=True),
        "Fiscalité (%)": st.column_config.NumberColumn("Fiscalité (%)", step=0.5, format="%.2f"),
    },
)

# Mémoriser les changements immédiatement si auto_apply
def apply_from_editors():
    # Synchroniser les DataFrames en session avec l'état retourné par les éditeurs
    st.session_state.pl_table = pl_editor.copy()
    st.session_state.per_table = per_editor.copy()
    # Reconstruire le modèle placements pour les calculs/exports
    st.session_state.placements = reconstruct_placements_from_tables(st.session_state.pl_table, st.session_state.per_table)

st.subheader("Périodes (édition directe)")
# Coherence: uid_placement doit référencer un uid de pl_table
if st.session_state.per_table.empty:
    st.session_state.per_table = pd.DataFrame(columns=["uid_placement","Début","Fin","Taux (%)"])

per_editor = st.data_editor(
    st.session_state.per_table,
    key="per_editor",
    use_container_width=True,
    num_rows="dynamic",
    hide_index=True,
    column_config={
        "uid_placement": st.column_config.TextColumn("uid_placement", help="Copiez-collez un uid de la table Placements"),
        "Début": st.column_config.TextColumn("Début", help="YYYY-MM-DD"),
        "Fin": st.column_config.TextColumn("Fin", help="YYYY-MM-DD"),
        "Taux (%)": st.column_config.NumberColumn("Taux (%)", step=0.05, format="%.3f"),
    },
)

if auto_apply:
    apply_from_editors()
else:
    st.button("Appliquer modifications", on_click=apply_from_editors, use_container_width=True)

# =========================================================
# Calculs et rendus
# =========================================================
st.divider()
st.subheader("Placements (synthèse)")
if not st.session_state.placements:
    st.info("Aucun placement. Éditez/ajoutez dans les tableaux ci-dessus.")
else:
    synth_rows=[]
    for p in st.session_state.placements:
        synth_rows.append({
            "Nom": p["nom"],
            "Somme (€)": p["somme"],
            "Taux défaut (%)": p["taux"],
            "Fiscalité (%)": p["fiscalite"]["taux"],
            "Nb périodes": len(p.get("periodes",[]))
        })
    st.dataframe(pd.DataFrame(synth_rows), use_container_width=True, hide_index=True)

    debut = st.session_state.periode_globale["debut"]
    fin = st.session_state.periode_globale["fin"]
    base_jour = int(base_jour)

    monthly_list=[]
    for _, pl_row in st.session_state.pl_table.iterrows():
        dfp = build_monthly_schedule_from_tables(pl_row, st.session_state.per_table, debut, fin, base_jour=base_jour)
        monthly_list.append(dfp)
    monthly = pd.concat(monthly_list, ignore_index=True) if monthly_list else pd.DataFrame(columns=[
        'Placement','Date','Capital','Taux(%)','Jours_pondérés','Int_brut','Int_net','nb_jours','Brut_moyen_jour','Net_moyen_jour'
    ])
    if not monthly.empty:
        monthly['Date'] = pd.to_datetime(monthly['Date'], errors='coerce')

    st.markdown("### Résultats mensuels")
    if not monthly.empty:
        md = monthly.sort_values(['Date','Placement']).copy()
        md['Date'] = pd.to_datetime(md['Date'], errors='coerce')
        md_fmt = md.copy(); md_fmt['Date']=md_fmt['Date'].dt.strftime("%Y-%m")
        st.dataframe(md_fmt[['Date','Placement','Capital','Taux(%)','nb_jours','Jours_pondérés','Int_brut','Int_net','Brut_moyen_jour','Net_moyen_jour']], use_container_width=True, hide_index=True)
    else:
        st.info("Aucune donnée mensuelle sur la période sélectionnée.")

    st.markdown("### Totaux par placement")
    if not monthly.empty:
        totals = monthly.groupby('Placement',as_index=False).agg({'Int_brut':'sum','Int_net':'sum','Capital':'first'}).rename(columns={
            'Int_brut':'Total brut (€)','Int_net':'Total net (€)','Capital':'Capital (€)'
        })
        totals['Total brut (€)']=totals['Total brut (€)'].round(2)
        totals['Total net (€)']=totals['Total net (€)'].round(2)
        st.dataframe(totals, use_container_width=True, hide_index=True)

        c1,c2,c3,c4 = st.columns(4)
        total_capital=float(totals['Capital (€)'].sum())
        total_brut=float(totals['Total brut (€)'].sum())
        total_net=float(totals['Total net (€)'].sum())
        c1.metric("Capital total", f"{total_capital:,.2f} €".replace(","," "))
        c2.metric("Intérêts bruts totaux", f"{total_brut:,.2f} €".replace(","," "))
        c3.metric("Intérêts nets totaux", f"{total_net:,.2f} €".replace(","," "))
        avg_net_rate = (total_net/total_capital*100.0) if total_capital>0 else 0.0
        c4.metric("Rendement net moyen", f"{avg_net_rate:.2f} %")
    else:
        totals = pd.DataFrame(columns=['Placement','Capital (€)','Total brut (€)','Total net (€)'])

    st.markdown("## Graphiques d’évolution (Plotly)")
    ms = monthly.sort_values('Date').copy()
    if not ms.empty:
        ms['Date']=pd.to_datetime(ms['Date'],errors='coerce')
        placements_list = ms['Placement'].unique().tolist()
        color_palette = px.colors.qualitative.Safe
        color_map = {plc: color_palette[i%len(color_palette)] for i, plc in enumerate(placements_list)}

        st.markdown("### Par placement - Intérêts mensuels (Brut vs Net)")
        for nom_pl in placements_list:
            dfp = ms[ms['Placement']==nom_pl].copy()
            fig = go.Figure()
            fig.add_trace(go.Bar(x=dfp['Date'], y=dfp['Int_brut'], name="Brut", marker_color=color_map[nom_pl],
                                 hovertemplate="Mois: %{x|%Y-%m}<br>Brut: %{y:.2f} €<extra></extra>"))
            fig.add_trace(go.Bar(x=dfp['Date'], y=dfp['Int_net'], name="Net", marker_color="rgba(0,0,0,0.45)",
                                 hovertemplate="Mois: %{x|%Y-%m}<br>Net: %{y:.2f} €<extra></extra>"))
            fig.update_layout(barmode='group', title_text=f"{nom_pl} - Intérêts mensuels", legend_title_text="Type",
                              xaxis_title="Mois", yaxis_title="€", margin=dict(t=50,l=40,r=20,b=40), height=360, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Cumul annuel par placement (Net)")
        cum_by_pl = ms.copy().sort_values(['Placement','Date'])
        cum_by_pl['Net_cumulé'] = cum_by_pl.groupby('Placement')['Int_net'].cumsum()
        fig_cumul = go.Figure()
        for nom_pl in placements_list:
            sub = cum_by_pl[cum_by_pl['Placement']==nom_pl]
            fig_cumul.add_trace(go.Scatter(x=sub['Date'], y=sub['Net_cumulé'], mode='lines+markers', name=nom_pl,
                                           line=dict(color=color_map[nom_pl], width=2), marker=dict(size=6),
                                           hovertemplate="Mois: %{x|%Y-%m}<br>Net cumulé: %{y:.2f} €<extra></extra>"))
        fig_cumul.update_layout(title_text="Évolution annuelle du net cumulé par placement", xaxis_title="Mois",
                                yaxis_title="€", legend_title_text="Placement", margin=dict(t=50,l=40,r=20,b=40),
                                height=420, template="plotly_white")
        st.plotly_chart(fig_cumul, use_container_width=True)

        st.markdown("### Cumul net global (aire empilée)")
        stacked = ms.copy(); stacked['Date_str']=stacked['Date'].dt.strftime("%Y-%m")
        pivot = stacked.pivot_table(index='Date_str', columns='Placement', values='Int_net', aggfunc='sum').fillna(0)
        pivot_cum = pivot.cumsum()
        fig_stack = go.Figure()
        for nom_pl in placements_list:
            fig_stack.add_trace(go.Scatter(x=pivot_cum.index, y=pivot_cum[nom_pl], mode='lines', name=nom_pl,
                                           line=dict(color=color_map[nom_pl], width=0.8), stackgroup='one',
                                           hovertemplate="Mois: %{x}<br>Cumul net: %{y:.2f} €<extra></extra>"))
        fig_stack.update_layout(title_text="Cumul net global (aire empilée)", xaxis_title="Mois", yaxis_title="€",
                                legend_title_text="Placement", margin=dict(t=50,l=40,r=20,b=40), height=420,
                                template="plotly_white")
        st.plotly_chart(fig_stack, use_container_width=True)
    else:
        st.info("Aucune donnée pour tracer les graphiques sur la période choisie.")

# =========================================================
# Export des résultats
# =========================================================
st.markdown("## Export des résultats")
if 'monthly' in globals():
    pass
# Export table mensuelle
if 'monthly' in locals() and not monthly.empty:
    csv_monthly = monthly.sort_values(['Date','Placement']).copy()
    csv_monthly['Date'] = pd.to_datetime(csv_monthly['Date'], errors='coerce').dt.strftime("%Y-%m-%d")
    csv_bytes = csv_monthly.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les résultats mensuels (CSV)", data=csv_bytes,
                       file_name="resultats_mensuels.csv", mime="text/csv")
else:
    st.caption("Export mensuel indisponible: aucune donnée.")

# Export totaux
if 'totals' in locals() and not totals.empty:
    csv_totals_bytes = totals.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les totaux par placement (CSV)", data=csv_totals_bytes,
                       file_name="totaux_par_placement.csv", mime="text/csv")
else:
    st.caption("Export totaux indisponible: aucune donnée.")

# Notes
st.divider()
with st.expander("Notes & limites"):
    st.markdown(
        "- Les tableaux Placements et Périodes affichés SONT la source de vérité; vos éditions y sont conservées en session.\n"
        "- Le CSV importé ne réécrit les tableaux qu'au moment de l'import. Après import, modifiez et cliquez sur « Appliquer modifications » (ou activez l'option automatique).\n"
        "- Pour lier une période à un placement, utilisez la colonne uid_placement; l'uid du placement est visible dans la colonne uid de la table Placements.\n"
        "- Calcul des intérêts: prorata linéaire base 365 par défaut; fiscalité simple appliquée sur les intérêts.\n"
    )
