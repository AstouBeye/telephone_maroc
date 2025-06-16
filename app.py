import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go # Pour le pie chart, si px.pie ne suffit pas (ici px.pie est suffisant)

st.set_page_config(layout="wide") # Utiliser toute la largeur de la page

st.title("Visualisation de ma base de données 'telephone_maroc' avec Plotly")

# Charger les données
df = pd.read_csv("telephones_maroc.csv")

# Nettoyer les données en supprimant les lignes avec des valeurs manquantes
df = df.dropna().reset_index(drop=True)

st.subheader("Aperçu de ma base de données")
st.dataframe(df)

st.subheader("Informations sur les colonnes")
st.write(df.columns.tolist())
st.write(df.describe())

# Section pour la sélection de colonne et de ville
st.subheader("Analyse personnalisée")

colonne = st.selectbox("Choisissez une colonne pour analyse", df.columns)
st.write(f"Vous avez choisi la colonne : **{colonne}**")

if 'Ville_vente' in df.columns:
    ville_selection = st.selectbox("Choisissez une ville pour filtrer", df["Ville_vente"].unique())
    st.write(f"Vous avez choisi la ville : **{ville_selection}**")

    st.subheader(f"Données filtrées pour {ville_selection}")
    df_ville = df[df["Ville_vente"] == ville_selection]
    st.dataframe(df_ville)

    # --- Premier graphique : Nombre de téléphones vendus par ville ---
    st.subheader("Nombre de téléphones vendus par ville")
    vente_par_ville = df["Ville_vente"].value_counts().reset_index()
    vente_par_ville.columns = ['Ville', 'Nombre de ventes'] # Renommer les colonnes pour Plotly

    fig_ville = px.bar(vente_par_ville, x='Ville', y='Nombre de ventes',
                       title="Téléphones vendus par ville",
                       labels={'Ville': 'Ville', 'Nombre de ventes': 'Nombre de ventes'})
    st.plotly_chart(fig_ville, use_container_width=True) # Afficher la figure Plotly dans Streamlit

else:
    st.warning("La colonne 'Ville_vente' n'existe pas dans votre fichier CSV. Certaines visualisations ne seront pas disponibles.")

# --- Deuxième ensemble de graphiques : Multi-plots avec Plotly ---
st.subheader("Analyses approfondies des caractéristiques des téléphones")

# Utilisation de st.columns pour organiser les graphiques côte à côte
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

with col1:
    # 1. Boxplot pour Autonomie_batterie
    if 'Autonomie_batterie' in df.columns:
        fig_autonomie = px.box(df, y='Autonomie_batterie',
                               title='Boxplot de l’Autonomie de la Batterie',
                               labels={'Autonomie_batterie': 'Autonomie Batterie (mAh)'})
        st.plotly_chart(fig_autonomie, use_container_width=True)
    else:
        st.info("Colonne 'Autonomie_batterie' manquante.")

with col2:
    # 2. Barplot pour Nb_coeurs
    if 'Nb_coeurs' in df.columns:
        fig_coeurs = px.bar(df, x='Nb_coeurs',
                            title='Distribution du Nombre de Cœurs',
                            labels={'Nb_coeurs': 'Nombre de Cœurs'})
        st.plotly_chart(fig_coeurs, use_container_width=True)
    else:
        st.info("Colonne 'Nb_coeurs' manquante.")

with col3:
    # 3. Pie chart pour Marque
    if 'Marque' in df.columns:
        fig_marque = px.pie(df, names='Marque',
                            title='Répartition des Marques')
        st.plotly_chart(fig_marque, use_container_width=True)
    else:
        st.info("Colonne 'Marque' manquante.")

with col4:
    # 4. Scatter plot entre Prix et Stockage_interne
    if 'Stockage_interne' in df.columns and 'Prix' in df.columns:
        fig_scatter = px.scatter(df, x='Stockage_interne', y='Prix',
                                 title='Relation entre Prix et Stockage Interne',
                                 labels={'Stockage_interne': 'Stockage Interne (Go)', 'Prix': 'Prix (MAD)'},
                                 symbol_sequence=['circle']) # 'circle' pour des points circulaires
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("Colonnes 'Prix' ou 'Stockage_interne' manquantes.")

