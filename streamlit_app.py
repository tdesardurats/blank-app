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
# Configuration
# =========================
st.set_page_config(page_title="Livrets: Gestionnaire d'intérêts", page_icon="💰", layout="wide")

# CSS pour améliorer l'apparence
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("💰 Gestionnaire de Livrets d'Épargne")
st.markdown("### Interface complète pour gérer vos placements, calculs d'intérêts et visualisations")

# =========================
# Fonctions utilitaires
# =========================
CSV_HEADER = ["type_ligne", "nom", "somme", "taux_defaut", "fisc_type", "fisc_taux", 
              "periode_debut", "periode_fin", "periode_taux"]

def safe_float(x, default=0.0):
    try:
        return float(str(x).replace(",", "."))
    except:
        return default

def export_to_csv(placements, periode_globale):
    rows = []
    # META ligne
    rows.append({
        "type_ligne": "META", "nom": "", "somme": "", "taux_defaut": "", 
        "fisc_type": "", "fisc_taux": "",
        "periode_debut": periode_globale['debut'].strftime("%Y-%m-%d"),
        "periode_fin": periode_globale['fin'].strftime("%Y-%m-%d"), "periode_taux": ""
    })
    
    # PLACEMENT lignes
    for p in placements:
        periods = p.get('periodes', [])
        if not periods:
            rows.append({
                "type_ligne": "PLACEMENT", "nom": p['nom'], "somme": p['somme'],
                "taux_defaut": p['taux'], "fisc_type": p['fiscalite']['type'],
                "fisc_taux": p['fiscalite']['taux'], "periode_debut": "", 
                "periode_fin": "", "periode_taux": ""
            })
        else:
            for per in periods:
                rows.append({
                    "type_ligne": "PLACEMENT", "nom": p['nom'], "somme": p['somme'],
                    "taux_defaut": p['taux'], "fisc_type": p['fiscalite']['type'],
                    "fisc_taux": p['fiscalite']['taux'], "periode_debut": per['debut'],
                    "periode_fin": per['fin'], "periode_taux": per['taux']
                })
    
    return pd.DataFrame(rows, columns=CSV_HEADER).to_csv(index=False).encode("utf-8")

def import_from_csv(file_bytes):
    df = pd.read_csv(io.BytesIO(file_bytes), dtype=str).fillna("")
    
    # Période globale
    meta = df[df['type_ligne'] == 'META']
    if not meta.empty:
        m0 = meta.iloc[0]
        try:
            debut = datetime.strptime(m0['periode_debut'], "%Y-%m-%d").date()
            fin = datetime.strptime(m0['periode_fin'], "%Y-%m-%d").date()
        except:
            today = date.today()
            debut, fin = date(today.year, 1, 1), date(today.year, 12, 31)
    else:
        today = date.today()
        debut, fin = date(today.year, 1, 1), date(today.year, 12, 31)
    
    # Placements
    plc_rows = df[df['type_ligne'] == 'PLACEMENT']
    placements_dict = {}
    
    for _, r in plc_rows.iterrows():
        nom = r['nom'].strip()
        if not nom: continue
        
        if nom not in placements_dict:
            placements_dict[nom] = {
                "uid": str(uuid.uuid4()), "nom": nom,
                "somme": safe_float(r['somme']), "taux": safe_float(r['taux_defaut']),
                "fiscalite": {
                    "type": r['fisc_type'] if r['fisc_type'] in ("PFU", "PERSONNALISE") else "PFU",
                    "taux": safe_float(r['fisc_taux'], 30.0)
                }, "periodes": []
            }
        
        if r['periode_debut'] and r['periode_fin'] and r['periode_taux']:
            placements_dict[nom]["periodes"].append({
                "debut": r['periode_debut'], "fin": r['periode_fin'],
                "taux": safe_float(r['periode_taux'])
            })
    
    return list(placements_dict.values()), {"debut": debut, "fin": fin}

# Fonctions de calcul (simplifiées)
def calc_interests(placements, periode_globale, base_jour=365):
    if not placements:
        return pd.DataFrame()
    
    def mois_range(start, end):
        current = date(start.year, start.month, 1)
        last = date(end.year, end.month, 1)
        months = []
        while current <= last:
            months.append(current)
            current = current + relativedelta(months=1)
        return months
    
    def days_in_month(d):
        return (d + relativedelta(months=1) - d).days
    
    results = []
    debut, fin = periode_globale['debut'], periode_globale['fin']
    
    for placement in placements:
        capital = placement['somme']
        tax_rate = placement['fiscalite']['taux']
        periods = placement.get('periodes', [])
        
        if not periods:
            periods = [{'debut': debut.strftime("%Y-%m-%d"), 'fin': fin.strftime("%Y-%m-%d"), 'taux': placement['taux']}]
        
        for month_start in mois_range(debut, fin):
            month_days = days_in_month(month_start)
            month_end = month_start + relativedelta(months=1) - relativedelta(days=1)
            
            total_brut = 0
            for period in periods:
                try:
                    p_start = max(datetime.strptime(period['debut'], "%Y-%m-%d").date(), month_start)
                    p_end = min(datetime.strptime(period['fin'], "%Y-%m-%d").date(), month_end)
                    if p_end >= p_start:
                        days = (p_end - p_start).days + 1
                        daily_rate = period['taux'] / 100 / base_jour
                        total_brut += capital * daily_rate * days
                except:
                    continue
            
            total_net = total_brut * (1 - tax_rate / 100)
            
            results.append({
                'Placement': placement['nom'], 'Date': month_start.strftime("%Y-%m"),
                'Capital': capital, 'Int_brut': round(total_brut, 2), 'Int_net': round(total_net, 2)
            })
    
    return pd.DataFrame(results)

# =========================
# État de session
# =========================
if 'placements' not in st.session_state:
    st.session_state.placements = []
if 'periode_globale' not in st.session_state:
    today = date.today()
    st.session_state.periode_globale = {'debut': date(today.year, 1, 1), 'fin': date(today.year, 12, 31)}

# =========================
# Interface principale avec onglets
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["📁 Import/Export", "💼 Mes Placements", "📊 Résultats & Graphiques", "⚙️ Paramètres"])

with tab4:
    st.header("⚙️ Paramètres Généraux")
    
    col1, col2 = st.columns(2)
    with col1:
        new_debut = st.date_input("🗓️ Date de début", 
                                  value=st.session_state.periode_globale['debut'])
    with col2:
        new_fin = st.date_input("🗓️ Date de fin", 
                                value=st.session_state.periode_globale['fin'])
    
    if new_debut <= new_fin:
        st.session_state.periode_globale = {'debut': new_debut, 'fin': new_fin}
    else:
        st.error("❌ La date de début doit être antérieure à la date de fin")
    
    st.markdown("---")
    base_jour = st.selectbox("📅 Base de calcul annuelle", [360, 365, 366], index=1)
    
    st.info("ℹ️ Ces paramètres s'appliquent à tous les calculs d'intérêts")

with tab1:
    st.header("📁 Import / Export des données")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📥 Importer un fichier CSV")
        uploaded_file = st.file_uploader("Choisissez votre fichier CSV", type=["csv"])
        
        if uploaded_file:
            if st.button("🔄 Charger les données", type="primary"):
                try:
                    placements_imported, periode_imported = import_from_csv(uploaded_file.read())
                    st.session_state.placements = placements_imported
                    st.session_state.periode_globale = periode_imported
                    st.success(f"✅ {len(placements_imported)} placement(s) importé(s) avec succès!")
                except Exception as e:
                    st.error(f"❌ Erreur lors de l'import: {str(e)}")
    
    with col2:
        st.subheader("📤 Exporter les données")
        if st.session_state.placements:
            csv_data = export_to_csv(st.session_state.placements, st.session_state.periode_globale)
            st.download_button(
                "💾 Télécharger CSV d'entrée", 
                data=csv_data, 
                file_name="mes_placements.csv",
                mime="text/csv",
                type="primary"
            )
            st.info(f"📋 {len(st.session_state.placements)} placement(s) prêt(s) à l'export")
        else:
            st.info("ℹ️ Aucun placement à exporter")

with tab2:
    st.header("💼 Gestion des Placements")
    
    # Ajout d'un nouveau placement
    with st.expander("➕ Ajouter un nouveau placement", expanded=len(st.session_state.placements)==0):
        with st.form("nouveau_placement"):
            col1, col2 = st.columns(2)
            with col1:
                nom = st.text_input("📝 Nom du placement", placeholder="ex: Livret A")
                somme = st.number_input("💰 Montant (€)", min_value=0.0, step=100.0, format="%.2f")
            with col2:
                taux = st.number_input("📈 Taux annuel (%)", min_value=0.0, step=0.1, format="%.2f")
                fisc_type = st.selectbox("🏛️ Fiscalité", ["PFU", "PERSONNALISE"])
                fisc_taux = st.number_input("💸 Taux fiscal (%)", 
                                           value=30.0 if fisc_type=="PFU" else 0.0, 
                                           step=0.5, format="%.1f")
            
            st.markdown("**📅 Périodes spécifiques (optionnel)**")
            col3, col4, col5 = st.columns(3)
            with col3:
                p_debut = st.date_input("Début période", value=st.session_state.periode_globale['debut'])
            with col4:
                p_fin = st.date_input("Fin période", value=st.session_state.periode_globale['fin'])
            with col5:
                p_taux = st.number_input("Taux période (%)", min_value=0.0, step=0.1, format="%.2f")
            
            submitted = st.form_submit_button("✅ Ajouter ce placement", type="primary")
            
            if submitted and nom:
                periodes = []
                if p_taux > 0 and p_debut <= p_fin:
                    periodes.append({
                        "debut": p_debut.strftime("%Y-%m-%d"),
                        "fin": p_fin.strftime("%Y-%m-%d"),
                        "taux": float(p_taux)
                    })
                
                nouveau_placement = {
                    "uid": str(uuid.uuid4()), "nom": nom, "somme": float(somme),
                    "taux": float(taux), "fiscalite": {"type": fisc_type, "taux": float(fisc_taux)},
                    "periodes": periodes
                }
                st.session_state.placements.append(nouveau_placement)
                st.success(f"✅ Placement '{nom}' ajouté avec succès!")
                st.rerun()
    
    # Liste des placements existants
    if st.session_state.placements:
        st.markdown("### 📋 Vos placements actuels")
        
        for i, placement in enumerate(st.session_state.placements):
            with st.expander(f"💼 {placement['nom']} - {placement['somme']:,.0f}€ à {placement['taux']:.2f}%"):
                
                # Modification du placement
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    new_nom = st.text_input("Nom", value=placement['nom'], key=f"nom_{i}")
                    new_somme = st.number_input("Montant (€)", value=float(placement['somme']), 
                                               step=100.0, format="%.2f", key=f"somme_{i}")
                
                with col2:
                    new_taux = st.number_input("Taux (%)", value=float(placement['taux']), 
                                              step=0.1, format="%.2f", key=f"taux_{i}")
                    new_fisc_type = st.selectbox("Fiscalité", ["PFU", "PERSONNALISE"], 
                                                 index=0 if placement['fiscalite']['type']=="PFU" else 1,
                                                 key=f"fisc_type_{i}")
                
                with col3:
                    new_fisc_taux = st.number_input("Taux fiscal (%)", 
                                                    value=float(placement['fiscalite']['taux']),
                                                    step=0.5, format="%.1f", key=f"fisc_taux_{i}")
                
                # Gestion des périodes
                st.markdown("**📅 Périodes spécifiques:**")
                if placement.get('periodes'):
                    periods_df = pd.DataFrame(placement['periodes'])
                    st.dataframe(periods_df, use_container_width=True, hide_index=True)
                    
                    if st.button("🗑️ Supprimer toutes les périodes", key=f"clear_periods_{i}"):
                        st.session_state.placements[i]['periodes'] = []
                        st.success("✅ Périodes supprimées")
                        st.rerun()
                else:
                    st.info("Aucune période spécifique - utilise le taux par défaut")
                
                # Boutons d'action
                col_a, col_b, col_c = st.columns([1, 1, 1])
                
                with col_a:
                    if st.button("💾 Sauvegarder", key=f"save_{i}", type="primary"):
                        st.session_state.placements[i].update({
                            "nom": new_nom, "somme": float(new_somme), "taux": float(new_taux),
                            "fiscalite": {"type": new_fisc_type, "taux": float(new_fisc_taux)}
                        })
                        st.success("✅ Modifications sauvegardées!")
                        st.rerun()
                
                with col_c:
                    if st.button("🗑️ Supprimer", key=f"delete_{i}", type="secondary"):
                        if st.session_state.get(f"confirm_delete_{i}"):
                            del st.session_state.placements[i]
                            st.success("✅ Placement supprimé!")
                            st.rerun()
                        else:
                            st.session_state[f"confirm_delete_{i}"] = True
                            st.warning("⚠️ Cliquez à nouveau pour confirmer")
    
    else:
        st.info("📝 Aucun placement enregistré. Utilisez le formulaire ci-dessus pour en ajouter un.")

with tab3:
    st.header("📊 Résultats et Visualisations")
    
    if not st.session_state.placements:
        st.info("ℹ️ Ajoutez des placements pour voir les résultats")
    else:
        # Calcul des résultats
        results_df = calc_interests(st.session_state.placements, st.session_state.periode_globale, base_jour)
        
        if not results_df.empty:
            # Métriques générales
            total_capital = sum(p['somme'] for p in st.session_state.placements)
            total_brut = results_df['Int_brut'].sum()
            total_net = results_df['Int_net'].sum()
            rendement_net = (total_net / total_capital * 100) if total_capital > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("💰 Capital Total", f"{total_capital:,.0f} €")
            col2.metric("📈 Intérêts Bruts", f"{total_brut:,.2f} €")
            col3.metric("💵 Intérêts Nets", f"{total_net:,.2f} €")
            col4.metric("📊 Rendement Net", f"{rendement_net:.2f}%")
            
            # Tableaux des résultats
            st.markdown("### 📋 Résultats détaillés")
            
            # Par placement
            totaux_placement = results_df.groupby('Placement').agg({
                'Capital': 'first', 'Int_brut': 'sum', 'Int_net': 'sum'
            }).round(2).reset_index()
            st.dataframe(totaux_placement, use_container_width=True, hide_index=True)
            
            # Résultats mensuels
            with st.expander("📅 Détail mensuel"):
                st.dataframe(results_df, use_container_width=True, hide_index=True)
            
            # Graphiques
            st.markdown("### 📈 Visualisations")
            
            # Graphique en barres par placement
            fig_bar = px.bar(totaux_placement, x='Placement', y=['Int_brut', 'Int_net'], 
                            title="Intérêts par placement", barmode='group')
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # Évolution mensuelle
            if len(results_df['Date'].unique()) > 1:
                monthly_pivot = results_df.pivot_table(
                    index='Date', columns='Placement', values='Int_net', aggfunc='sum'
                ).fillna(0)
                
                fig_line = go.Figure()
                for placement in monthly_pivot.columns:
                    fig_line.add_trace(go.Scatter(
                        x=monthly_pivot.index, y=monthly_pivot[placement].cumsum(),
                        mode='lines+markers', name=placement
                    ))
                
                fig_line.update_layout(
                    title="Évolution cumulative des intérêts nets",
                    xaxis_title="Mois", yaxis_title="Intérêts cumulés (€)", height=400
                )
                st.plotly_chart(fig_line, use_container_width=True)
            
            # Export des résultats
            st.markdown("### 💾 Export des résultats")
            col1, col2 = st.columns(2)
            
            with col1:
                csv_results = results_df.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Résultats mensuels (CSV)", 
                                 data=csv_results, file_name="resultats_mensuels.csv")
            
            with col2:
                csv_totaux = totaux_placement.to_csv(index=False).encode('utf-8')
                st.download_button("📥 Totaux par placement (CSV)", 
                                 data=csv_totaux, file_name="totaux_placements.csv")
        
        else:
            st.warning("⚠️ Aucun résultat calculable avec les paramètres actuels")

# Footer
st.markdown("---")
st.markdown("*💡 Interface développée pour une gestion simplifiée de vos livrets d'épargne*")
