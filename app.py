import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Visualisation de ma base telephone_maroc")
df = pd.read_csv("telephones_maroc.csv")
st.subheader("Aperçu de ma base")
st.dataframe(df)
st.subheader("Nom des colonnes")
st.write(df.columns.tolist())
st.write(df.describe())
colonne = st.selectbox("choisir une colonne", df.columns)
ville_selection = st.selectbox("choisir une ville", df.Ville_vente)
col_ville = "Ville_vente"
df_ville = df[df[col_ville] == ville_selection]
df =df.dropna()
vente_par_ville = df["Ville_vente"].value_counts()
st.subheader("Nombre de telephones vendus par ville")
fig, ax = plt.subplots()
vente_par_ville.plot(kind = "bar", ax=ax)
ax.set_xlabel("Ville")
ax.set_ylabel("Nombre de ventes")
ax.set_title("Telephones vendusnpar ville")
st.pyplot(fig)

# Configuration du style
sns.set(style="whitegrid")

# Création des sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Boxplot pour Autonomie_batterie
sns.boxplot(data=df, y='Autonomie_batterie', ax=axes[0, 0], color='skyblue')
axes[0, 0].set_title('Boxplot de l’Autonomie de la Batterie')

# 2. Barplot pour Nb_coeurs
sns.countplot(data=df, x='Nb_coeurs', ax=axes[0, 1], palette='muted')
axes[0, 1].set_title('Barplot du Nombre de Cœurs')

# 3. Pie chart pour Marque
marque_counts = df['Marque'].value_counts()
axes[1, 0].pie(marque_counts, labels=marque_counts.index, autopct='%1.1f%%', startangle=140)
axes[1, 0].set_title('Répartition des Marques')

# 4. Scatter plot (points carrés) entre Prix et Stockage_interne
sns.scatterplot(data=df, x='Stockage_interne', y='Prix', ax=axes[1, 1], marker='s', color='green')
axes[1, 1].set_title('Square Plot entre Prix et Stockage Interne')

# Ajuster l’affichage
plt.tight_layout()
plt.show()

