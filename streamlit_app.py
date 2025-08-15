import streamlit as st
import pandas as pd

st.set_page_config(page_title="Calcul d'intérêts - Livrets", page_icon="💶", layout="centered")

st.title("💶 Visualiseur d'intérêts pour livrets bancaires")
st.caption("Saisir plusieurs placements, puis visualiser les intérêts journaliers, mensuels et annuels.")

# État initial du tableau de placements
if "placements" not in st.session_state:
    st.session_state.placements = pd.DataFrame(columns=["Nom", "Somme (€)", "Taux annuel (%)"])

def ajouter_placement(nom, somme, taux):
    new_row = {"Nom": nom.strip(), "Somme (€)": somme, "Taux annuel (%)": taux}
    st.session_state.placements = pd.concat(
        [st.session_state.placements, pd.DataFrame([new_row])],
        ignore_index=True
    )

def calcul_interets(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    # Taux en décimal
    taux_dec = out["Taux annuel (%)"] / 100.0
    interet_annuel = out["Somme (€)"] * taux_dec
    interet_journalier = interet_annuel / 365.0
    interet_mensuel = interet_annuel / 12.0
    out["Intérêt/jour (€)"] = interet_journalier.round(4)
    out["Intérêt/mois (€)"] = interet_mensuel.round(2)
    out["Intérêt/an (€)"] = interet_annuel.round(2)
    return out

with st.form("form_placement", clear_on_submit=True):
    st.subheader("Ajouter un placement")
    col1, col2, col3 = st.columns(3)
    with col1:
        nom = st.text_input("Nom du placement", placeholder="Livret A")
    with col2:
        somme = st.number_input("Somme investie (€)", min_value=0.0, step=100.0, format="%.2f")
    with col3:
        taux = st.number_input("Taux annuel (%)", min_value=0.0, step=0.1, format="%.3f")

    submitted = st.form_submit_button("Ajouter")
    if submitted:
        if not nom:
            st.warning("Veuillez saisir un nom de placement.")
        else:
            ajouter_placement(nom, somme, taux)
            st.success(f"Placement « {nom} » ajouté.")

st.divider()

st.subheader("Placements saisis")
if st.session_state.placements.empty:
    st.info("Aucun placement pour l’instant. Ajoutez-en via le formulaire ci-dessus.")
else:
    # Édition directe dans le tableau si besoin
    st.caption("Astuce: les valeurs du tableau sont modifiables. Cliquez pour éditer.")
    edited_df = st.data_editor(
        st.session_state.placements,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Somme (€)": st.column_config.NumberColumn("Somme (€)", step=50.0, format="%.2f"),
            "Taux annuel (%)": st.column_config.NumberColumn("Taux annuel (%)", step=0.05, format="%.3f"),
        },
        key="editor",
    )
    # Synchronisation avec l'état
    st.session_state.placements = edited_df

    # Calculs
    resultats = calcul_interets(st.session_state.placements)

    st.subheader("Résultats calculés")
    st.dataframe(
        resultats[["Nom", "Somme (€)", "Taux annuel (%)", "Intérêt/jour (€)", "Intérêt/mois (€)", "Intérêt/an (€)"]],
        use_container_width=True,
        hide_index=True
    )

    # Totaux
    totaux = {
        "Total Sommes (€)": float(resultats["Somme (€)"].sum()) if not resultats.empty else 0.0,
        "Total Intérêts/jour (€)": float(resultats["Intérêt/jour (€)"].sum()) if "Intérêt/jour (€)" in resultats else 0.0,
        "Total Intérêts/mois (€)": float(resultats["Intérêt/mois (€)"].sum()) if "Intérêt/mois (€)" in resultats else 0.0,
        "Total Intérêts/an (€)": float(resultats["Intérêt/an (€)"].sum()) if "Intérêt/an (€)" in resultats else 0.0,
    }
    st.markdown("### Totaux")
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Sommes", f"{totaux['Total Sommes (€)']:.2f} €")
    colB.metric("Intérêts/jour", f"{totaux['Total Intérêts/jour (€)']:.2f} €")
    colC.metric("Intérêts/mois", f"{totaux['Total Intérêts/mois (€)']:.2f} €")
    colD.metric("Intérêts/an", f"{totaux['Total Intérêts/an (€)']:.2f} €")

    # Export CSV
    csv = resultats.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Télécharger les résultats (CSV)", data=csv, file_name="resultats_livrets.csv", mime="text/csv")

st.divider()

with st.expander("Options & précisions"):
    st.markdown(
        "- Les intérêts sont calculés sur la base d’un taux annuel simple, réparti linéairement par jour (365) et par mois (12).\n"
        "- Pour les livrets réglementés français (Livret A, LDDS, etc.), les banques appliquent la règle des quinzaines et des variations de taux dans l’année. "
        "Cette application simplifie en utilisant un prorata journalier. Si besoin, je peux ajouter la règle des quinzaines et la gestion de périodes avec taux changeant."
    )
