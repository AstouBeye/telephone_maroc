import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Chargement des données
df = pd.read_csv("telephones_maroc.csv")

# Titre de l'application
st.title("📊 Régression Linéaire : Prix des Téléphones au Maroc")

# Affichage du dataset
if st.checkbox("Afficher les données brutes"):
    st.write(df)

# Sélection des variables
X = df[['Stockage_interne', 'Charge_rapide', 'Autonomie_batterie']]
y = df['Prix']

# Traitement des données manquantes
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    st.error("⚠️ Données manquantes détectées. Veuillez les corriger.")
    st.stop()

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Évaluation
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("🧮 Résultats de la régression")
st.write(f"**RMSE :** {mse ** 0.5:.2f}")
st.write(f"**R² :** {r2:.2f}")
st.write(f"**Intercept :** {model.intercept_:.2f}")
st.write("**Coefficients :**")
coeffs = pd.DataFrame({
    'Variable': X.columns,
    'Coefficient': model.coef_
})
st.table(coeffs)

# Visualisation du prix réel vs prédit
st.subheader("📈 Graphique : Prix réel vs Prix prédit")
fig, ax = plt.subplots()
sns.scatterplot(x=y_test, y=y_pred, ax=ax)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
ax.set_xlabel("Prix Réel")
ax.set_ylabel("Prix Prédit")
ax.set_title("Prix Réel vs Prix Prédit")
st.pyplot(fig)

# Section prédiction interactive
st.subheader("🔍 Prédiction du prix d’un téléphone")

stockage = st.slider("Stockage Interne (Go)", 32, 512, 128, step=32)
charge = st.selectbox("Charge Rapide", [0, 1])
autonomie = st.slider("Autonomie Batterie (heures)", 10, 72, 36)

donnees = pd.DataFrame([{
    'Stockage_interne': stockage,
    'Charge_rapide': charge,
    'Autonomie_batterie': autonomie
}])

prix_pred = model.predict(donnees)[0]
st.success(f"💰 Prix estimé : {prix_pred:.2f} MAD")
