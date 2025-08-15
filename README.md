# 💶 Visualiseur d'intérêts pour livrets bancaires

Application Streamlit permettant d’ajouter plusieurs placements (nom, somme, taux) et de visualiser les intérêts **par jour**, **par mois** et **par an**, avec totaux et export CSV. Optimisée pour un usage mobile (iPhone).

## Déploiement sur Streamlit Cloud
1. Créer un dépôt GitHub contenant ces fichiers: `app.py`, `requirements.txt`, `.streamlit/config.toml`.
2. Aller sur Streamlit Community Cloud et cliquer “Deploy an app”.
3. Sélectionner le dépôt, la branche et le fichier principal `app.py`.
4. Lancer le déploiement; une URL publique sera fournie.

## Utilisation sur iPhone
- Ouvrir l’URL publique dans Safari.
- Astuce: “Partager” > “Sur l’écran d’accueil” pour un accès façon application.
- Les données restent en session du navigateur; utiliser le bouton CSV pour exporter.

## Fonctionnalités
- Ajout et édition de placements.
- Calcul automatique: intérêt/jour, intérêt/mois, intérêt/an.
- Totaux agrégés.
- Export CSV.

## Limites et améliorations
- Calcul simplifié (prorata linéaire).  
- Possibles améliorations:
  - Règle des quinzaines (Livret A/LDDS).
  - Taux variables dans l’année.
  - Fiscalité (PFU, PS) et affichage du net.
  - Sauvegarde sur fichier/cloud.
