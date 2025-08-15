# 💶 Visualiseur d'intérêts pour livrets bancaires
Application Streamlit pour gérer plusieurs placements de type livret (nom, capital, fiscalité, périodes de taux), calculer les intérêts **bruts** et **nets**, visualiser l'évolution mensuelle et annuelle, et exporter/réimporter les données d’entrée.

## ✨ Fonctionnalités
- Saisie de plusieurs placements (nom, somme investie).
- Fiscalité par placement (PFU 30% ou personnalisé), calcul du **net**.
- Périodes de taux variables dans l’année (une ou plusieurs périodes par placement).
- Calcul mensuel avec base jours/an configurable (365 par défaut).
- Tableau mensuel enrichi:
  - nb_jours (taille du mois)
  - Jours_pondérés (jours couverts par les périodes)
  - Int_brut / Int_net
  - Moyennes par jour (Brut_moyen_jour / Net_moyen_jour)
- Totaux par placement et indicateur de rendement net moyen.
- Graphiques Plotly interactifs:
  - Intérêts mensuels (Brut vs Net) par placement (barres)
  - Cumul mensuel net par placement (lignes)
  - Cumul annuel net par placement (couleur par placement)
  - Cumul net global en aire empilée
- Export des **résultats** (mensuel et totaux) en CSV.
- Export/Import des **données d’entrée** en CSV pour reprendre plus tard (placements, fiscalité, périodes, période globale).

## 📦 Structure du dépôt
