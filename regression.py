import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Charger les données
df = pd.read_csv("telephones_maroc.csv")

# Sélection des variables
X = df[['Stockage_interne', 'Charge_rapide', 'Autonomie_batterie']]
y = df['Prix']

# Vérification des valeurs manquantes
if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
    print("⚠️ Données manquantes détectées. Veuillez nettoyer les données avant de continuer.")
    exit()

# Séparation en données d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Création du modèle
model = LinearRegression()
model.fit(X_train, y_train)

# Prédiction
y_pred = model.predict(X_test)

# Évaluation du modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Affichage des résultats
print("📊 Régression Linéaire Multiple")
print(f"Coefficients : {model.coef_}")
print(f"Intercept : {model.intercept_}")
print(f"Erreur quadratique moyenne (RMSE) : {mse ** 0.5:.2f}")
print(f"Coefficient de détermination (R²) : {r2:.2f}")

# Affichage graphique : prix réel vs prédit
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Prix Réel")
plt.ylabel("Prix Prédit")
plt.title("Prix Réel vs Prix Prédit")
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.grid(True)
plt.tight_layout()
plt.show()
