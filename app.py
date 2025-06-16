import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# boxplot pour la variable autonomie


