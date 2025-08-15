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
- Export/Import des **données d’entrée** en CSV pour reprendre plus tard (placements, fiscalité, périodes, période 


## ☁️ Déploiement sur Streamlit Community Cloud
1. Créer un dépôt GitHub avec `app.py`, `requirements.txt` et `.streamlit/config.toml`.
2. Aller sur Streamlit Community Cloud et cliquer “Create app”.
3. Sélectionner:
- Repository: votre dépôt
- Branch: main (ou celle utilisée)
- Main file path: `app.py`
4. Déployer; à la fin, une **URL publique** est fournie.

Chaque modification poussée sur GitHub redéploie automatiquement l’application.

## 📱 Utilisation sur iPhone
- Ouvrir l’URL publique dans Safari.
- “Partager” > “Sur l’écran d’accueil” pour créer un raccourci façon application.
- Les graphes Plotly sont interactifs (zoom, déplacement, tooltips).

## ▶️ Guide d’utilisation
1. Définir la **période globale** (barre latérale).
2. Ajouter un **placement**:
- Nom, Somme investie
- Taux annuel “défaut” (utilisé si aucune période n’est définie)
- Fiscalité (PFU 30% ou personnalisé)
- (Facultatif) Ajouter des **périodes de taux** (début, fin, taux)
- Enregistrer le placement
3. Visualiser:
- “Placements”: synthèse des réglages
- “Résultats mensuels enrichis”: tableau détaillé mois par mois
- “Totaux par placement” + métriques globales
- “Graphiques d’évolution (Plotly)”: par placement, cumul annuel, global empilé
4. Exporter:
- “Résultats mensuels (CSV)” et “Totaux par placement (CSV)”
- “Exporter les **données d’entrée** (CSV)” pour pouvoir reprendre plus tard
5. Reprendre plus tard:
- Utiliser “Importer un CSV de **données d’entrée**” et sélectionner le fichier précédemment exporté

## 🧾 Format du CSV d’entrée (export/import)
- 1 ligne `META` pour la période globale (`periode_debut`, `periode_fin`)
- 1..n lignes `PLACEMENT` avec:
- nom, somme, taux_defaut, fiscalité (type, taux)
- périodes optionnelles (periode_debut, periode_fin, periode_taux)
- Si un placement n’a pas de période, une seule ligne est exportée pour ce placement avec les colonnes période vides.


## ⚠️ Notes & limites
- Modèle d’intérêts: **prorata journalier linéaire** sur base jours/an configurable (par défaut 365).
- Fiscalité appliquée comme pourcentage unique sur les intérêts (modèle simplifié).
- Les **périodes de taux** permettent de modéliser des changements intra-annuels; sans période, le taux “défaut” s’applique à toute la période globale.
- Pour les livrets réglementés (Livret A/LDDS), il est possible d’implémenter la **règle des quinzaines** et/ou des intérêts composés (capitalisation) en extension.

## 🚀 Feuille de route (suggestions)
- Règle des quinzaines (Livret A/LDDS).
- Séparation Prélèvements sociaux / IR (PFU 12.8% + PS 17.2%) et gestion des exonérations.
- Capitalisation mensuelle/journalière des intérêts (intérêts composés).
- Sauvegarde automatique vers Google Sheets ou GitHub Gist.
- Thèmes clair/sombre synchronisés avec Plotly.

---
Fait pour une utilisation simple sur mobile et un déploiement rapide sur Streamlit Cloud.
