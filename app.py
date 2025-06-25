import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

st.set_page_config(layout="wide", page_title="Analyse des Téléphones au Maroc")

# --- Configuration et Chargement des Données ---
st.sidebar.title("Navigation")
section = st.sidebar.radio("Aller à", ["Aperçu des Données", "Exploration Visuelle", "Modélisation Prédictive"])

@st.cache_data # Mettre en cache les données pour de meilleures performances
def load_and_preprocess_data(file_path, handle_missing='drop'):
    try:
        df = pd.read_csv(file_path)
        
        # Convertir les noms de colonnes pour une meilleure manipulation (minuscules, underscores)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('-', '_').str.replace('(', '').str.replace(')', '').str.lower()
        
        # Identifier les colonnes numériques et catégorielles
        numeric_cols = df.select_dtypes(include=np.number).columns
        categorical_cols = df.select_dtypes(include='object').columns

        # Convertir les colonnes numériques en type approprié, en gérant les erreurs
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Gérer les valeurs manquantes
        initial_rows = df.shape[0]
        if handle_missing == 'drop':
            df = df.dropna().reset_index(drop=True)
            st.sidebar.info(f"Suppression des lignes avec valeurs manquantes : {initial_rows - df.shape[0]} lignes supprimées.")
        elif handle_missing == 'fill_median':
            for col in numeric_cols:
                if df[col].isnull().any():
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val)
            for col in categorical_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()[0]
                    df[col] = df[col].fillna(mode_val)
            st.sidebar.info("Valeurs manquantes remplies (médiane pour numérique, mode pour catégorielle).")
        
        return df
    except FileNotFoundError:
        st.error(f"Erreur : Le fichier '{file_path}' n'a pas été trouvé. Assurez-vous qu'il est dans le même répertoire que l'application Streamlit.")
        st.stop()
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement ou du prétraitement des données : {e}")
        st.stop()

# Options pour le traitement des données manquantes
missing_data_handle_option = st.sidebar.selectbox(
    "Comment gérer les données manquantes?",
    ("drop", "fill_median"),
    help="Choisir 'drop' pour supprimer les lignes avec des valeurs manquantes, ou 'fill_median' pour remplir avec la médiane (numérique) / mode (catégorielle)."
)

df = load_and_preprocess_data("telephones_maroc.csv", missing_data_handle_option)

# --- Section : Aperçu des Données ---
if section == "Aperçu des Données":
    st.title("📊 Aperçu de la Base de Données 'Telephones Maroc'")
    st.write("Cette application interactive vous aide à explorer et à analyser votre dataset de téléphones au Maroc.")

    st.subheader("Structure et Contenu du Dataset")
    st.dataframe(df.head())

    st.subheader("Informations Générales")
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.write(f"**Nombre de lignes :** {df.shape[0]}")
        st.write(f"**Nombre de colonnes :** {df.shape[1]}")
        st.write("**Types de données :**")
        st.write(df.dtypes)
    with col_info2:
        st.write("**Colonnes disponibles :**")
        st.write(df.columns.tolist())
        st.write("**Statistiques descriptives :**")
        st.write(df.describe().T)

    st.subheader("Valeurs Manquantes")
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    if not missing_data.empty:
        st.warning("Des valeurs manquantes ont été trouvées et traitées selon votre choix en barre latérale.")
        st.dataframe(missing_data.sort_values(ascending=False))
    else:
        st.success("Aucune valeur manquante détectée après le prétraitement.")

    st.subheader("Analyse de la Distribution des Valeurs Catégorielles")
    categorical_cols = df.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
     selected_cat_col = st.selectbox("Choisissez une colonne catégorielle à analyser :", categorical_cols)
    
    st.write(f"**Fréquence des valeurs pour '{selected_cat_col.replace('_', ' ').title()}' :**")
    # Crée un DataFrame temporaire pour les comptages
    counts_df = df[selected_cat_col].value_counts().reset_index()
    # Renomme les colonnes du DataFrame temporaire de manière explicite
    counts_df.columns = [selected_cat_col, 'Count'] # La première colonne sera le nom de la catégorie, la seconde 'Count'
    
    st.dataframe(counts_df) # Affiche le DataFrame des comptages pour vérification

    fig_cat = px.bar(counts_df, # Utilise le DataFrame renommé
                     x=selected_cat_col, # L'axe des X est maintenant le nom de la colonne catégorielle
                     y='Count',           # L'axe des Y est maintenant 'Count'
                     title=f"Distribution de {selected_cat_col.replace('_', ' ').title()}",
                     labels={selected_cat_col: selected_cat_col.replace('_', ' ').title(), 'Count': 'Nombre d\'occurrences'},
                     color_discrete_sequence=px.colors.qualitative.Pastel) # Ajoute une palette de couleurs pour un meilleur visuel
    st.plotly_chart(fig_cat, use_container_width=True)
else:
    st.info("Aucune colonne catégorielle trouvée dans le dataset pour cette analyse.")
# ... (Previous code) ...

# This 'if' statement must precede the 'elif'
if section == "Aperçu des Données":
    st.title("📊 Aperçu de la Base de Données 'Telephones Maroc'")
    st.write("Cette application interactive vous aide à explorer et à analyser votre dataset de téléphones au Maroc.")
    # ... (rest of the "Aperçu des Données" section) ...



# **Exploration Visuelle**

# This 'elif' should be at the same indentation level as the 'if' above
elif section == "Exploration Visuelle":
    st.title("📈 Exploration Visuelle des Données de Téléphones")
    # ... (rest of the "Exploration Visuelle" section) ...

    # --- Filtres Globaux pour les Visualisations ---
    st.sidebar.subheader("Filtres pour Visualisations")
    
    df_filtered = df.copy() # Initialiser df_filtered avec le DataFrame complet

    # Filtre par marque
    if 'marque' in df_filtered.columns:
        marques_uniques = ['Toutes'] + sorted(df_filtered['marque'].unique().tolist())
        selected_marque = st.sidebar.selectbox("Filtrer par Marque :", marques_uniques)
        if selected_marque != 'Toutes':
            df_filtered = df_filtered[df_filtered['marque'] == selected_marque]
    else:
        st.sidebar.info("La colonne 'marque' n'est pas disponible pour le filtrage.")

    # Filtre par ville de vente
    if 'ville_vente' in df_filtered.columns:
        villes_uniques = ['Toutes'] + sorted(df_filtered['ville_vente'].unique().tolist())
        selected_ville = st.sidebar.selectbox("Filtrer par Ville de Vente :", villes_uniques)
        if selected_ville != 'Toutes':
            df_filtered = df_filtered[df_filtered['ville_vente'] == selected_ville]
    else:
        st.sidebar.warning("La colonne 'ville_vente' n'existe pas pour le filtrage.")
    
    if df_filtered.empty:
        st.warning("Aucune donnée ne correspond à vos critères de filtre. Veuillez ajuster les filtres.")
        st.stop()


    st.subheader("1. Analyse par Ville et Marque")
    col1, col2 = st.columns(2)
    with col1:
        if 'ville_vente' in df_filtered.columns:
            st.write("#### Nombre de téléphones vendus par ville")
            vente_par_ville = df_filtered["ville_vente"].value_counts().reset_index()
            vente_par_ville.columns = ['Ville', 'Nombre de ventes']
            fig_ville_bar = px.bar(vente_par_ville, x='Ville', y='Nombre de ventes',
                                   title="Téléphones vendus par ville (filtré)",
                                   labels={'Ville': 'Ville', 'Nombre de ventes': 'Nombre de ventes'})
            st.plotly_chart(fig_ville_bar, use_container_width=True)
        else:
            st.info("Colonne 'ville_vente' manquante pour cette visualisation.")

    with col2:
        if 'marque' in df_filtered.columns:
            st.write("#### Répartition des Marques")
            fig_marque_pie = px.pie(df_filtered, names='marque',
                                    title='Répartition des Marques (filtrée)')
            st.plotly_chart(fig_marque_pie, use_container_width=True)
        else:
            st.info("Colonne 'marque' manquante pour cette visualisation.")

    st.subheader("2. Distributions des Caractéristiques Clés")
    col3, col4 = st.columns(2)
    with col3:
        if 'autonomie_batterie' in df_filtered.columns:
            st.write("#### Distribution de l'Autonomie de la Batterie")
            fig_autonomie_hist = px.histogram(df_filtered, x='autonomie_batterie',
                                              title='Distribution de l’Autonomie de la Batterie',
                                              labels={'autonomie_batterie': 'Autonomie Batterie (mAh)'},
                                              nbins=20)
            st.plotly_chart(fig_autonomie_hist, use_container_width=True)
        else:
            st.info("Colonne 'autonomie_batterie' manquante.")

    with col4:
        if 'stockage_interne' in df_filtered.columns:
            st.write("#### Distribution du Stockage Interne")
            fig_stockage_hist = px.histogram(df_filtered, x='stockage_interne',
                                             title='Distribution du Stockage Interne',
                                             labels={'stockage_interne': 'Stockage Interne (Go)'},
                                             nbins=20)
            st.plotly_chart(fig_stockage_hist, use_container_width=True)
        else:
            st.info("Colonne 'stockage_interne' manquante.")

    st.subheader("3. Relations entre les Variables Numériques")
    numeric_cols = df_filtered.select_dtypes(include=np.number).columns.tolist()
    if 'prix' in numeric_cols:
        numeric_cols.remove('prix') # Souvent la variable cible, donc on l'enlève des options X mais on la garde pour Y
    
    if len(numeric_cols) >= 1:
        col_scatter1, col_scatter2 = st.columns(2)
        with col_scatter1:
            st.write("#### Nuage de points personnalisé")
            # Assurer que 'prix' est une option valide pour l'axe Y si elle existe
            y_axis_options = ['prix'] + [col for col in numeric_cols if col != 'prix'] if 'prix' in df_filtered.columns else numeric_cols
            if not y_axis_options:
                st.warning("Pas de colonnes numériques disponibles pour l'axe Y.")
            else:
                x_axis = st.selectbox("Axe X (Numérique) :", numeric_cols, key='scatter_x_axis')
                y_axis = st.selectbox("Axe Y (Numérique) :", y_axis_options, key='scatter_y_axis')
                color_by_options = ['Aucun'] + df_filtered.select_dtypes(include='object').columns.tolist()
                color_by = st.selectbox("Colorer par (Optionnel) :", color_by_options, key='scatter_color_by')

                if x_axis and y_axis:
                    fig_custom_scatter = px.scatter(df_filtered, x=x_axis, y=y_axis,
                                                    color=color_by if color_by != 'Aucun' else None,
                                                    title=f'{y_axis.replace("_", " ").title()} vs {x_axis.replace("_", " ").title()}',
                                                    labels={x_axis: x_axis.replace("_", " ").title(), y_axis: y_axis.replace("_", " ").title()})
                    st.plotly_chart(fig_custom_scatter, use_container_width=True)
                else:
                    st.info("Veuillez sélectionner au moins un axe X et Y pour le nuage de points.")
    else:
        st.info("Pas assez de colonnes numériques pour un nuage de points personnalisé.")

    st.subheader("4. Carte de Chaleur des Corrélations")
    numeric_df = df_filtered.select_dtypes(include=np.number)
    if not numeric_df.empty and len(numeric_df.columns) > 1:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
        corr_matrix = numeric_df.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        ax_corr.set_title("Matrice de Corrélation des Variables Numériques")
        st.pyplot(fig_corr)
    else:
        st.info("Pas assez de variables numériques pour calculer la matrice de corrélation ou le dataset filtré est vide.")



# **Modélisation Prédictive**

elif section == "Modélisation Prédictive": # Cette ligne doit suivre directement le 'if' ou 'elif' précédent
      st.title("🤖 Modélisation du Prix des Téléphones (Régression Linéaire)")
      st.write("Cette section permet de construire un modèle de régression linéaire pour prédire le prix des téléphones.")
    # Affichage du dataset pour le modèle (optionnel)
if st.checkbox("Afficher les données utilisées pour la modélisation"):
        st.dataframe(df)

st.subheader("Configuration du Modèle")
    
    # Sélection de la variable cible
target_variable = 'prix'
if target_variable not in df.columns:
        st.error(f"La colonne cible '{target_variable}' est introuvable. Veuillez vérifier votre dataset.")
        st.stop()
    
   # Sélection des variables explicatives (features)
# Exclure la colonne cible et les colonnes non numériques/non pertinentes pour la régression
available_features = df.select_dtypes(include=np.number).columns.tolist() # Correction ici
if target_variable in available_features:
    available_features.remove(target_variable)

    # Ajouter les colonnes binaires (si elles existent et n'ont pas été traitées comme numériques)
    # Exemple pour 'charge_rapide' si elle est 0 ou 1
    if 'charge_rapide' in df.columns and df['charge_rapide'].nunique() <= 2:
         if 'charge_rapide' not in available_features: # Éviter les doublons si déjà numérique
             available_features.append('charge_rapide')


    selected_features = st.multiselect(
        "Sélectionnez les variables explicatives (features) :",
        options=available_features,
        default=[col for col in ['stockage_interne', 'autonomie_batterie', 'nb_coeurs', 'charge_rapide'] if col in available_features] or available_features[:3] # Défaut intelligent
    )

    if not selected_features:
        st.warning("Veuillez sélectionner au moins une variable explicative.")
        st.stop()

    X = df[selected_features]
    y = df[target_variable]

    # Traitement des données manquantes dans les colonnes sélectionnées (si non géré en amont)
    if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
        st.warning("⚠️ Des données manquantes ont été détectées dans les colonnes sélectionnées. Elles seront supprimées pour l'entraînement du modèle.")
        # Nettoyer spécifiquement X et y pour le modèle
        data_model = pd.concat([X, y], axis=1).dropna()
        X = data_model[selected_features]
        y = data_model[target_variable]
        if X.empty:
            st.error("Après suppression des NaN, le jeu de données pour la modélisation est vide. Ajustez vos sélections ou nettoyez mieux les données.")
            st.stop()
        
    st.info(f"Le modèle sera entraîné avec **{len(selected_features)}** variables explicatives et **{X.shape[0]}** observations.")

    # Séparation train/test
    test_size = st.slider("Taille du jeu de test (%)", 10, 50, 20, step=5) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    st.write(f"**Taille du jeu d'entraînement :** {X_train.shape[0]} observations")
    st.write(f"**Taille du jeu de test :** {X_test.shape[0]} observations")

    # Modèle
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Évaluation
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    st.subheader("📈 Résultats de la Régression Linéaire")
    st.write(f"**Erreur Quadratique Moyenne (MSE) :** `{mse:.2f}`")
    st.write(f"**Racine de l'Erreur Quadratique Moyenne (RMSE) :** `{rmse:.2f}`")
    st.write(f"**Coefficient de Détermination (R²) :** `{r2:.2f}`")
    st.write(f"**Ordonnée à l'origine (Intercept) :** `{model.intercept_:.2f}`")
    
    st.write("---")
    st.write("**Coefficients des Variables :**")
    coeffs = pd.DataFrame({
        'Variable': selected_features,
        'Coefficient': model.coef_
    })
    st.dataframe(coeffs)
    st.info("Un **coefficient positif** indique que l'augmentation de la variable associée tend à augmenter le prix, et vice-versa pour un coefficient négatif. La **magnitude** indique l'ampleur de cette influence.")

    # Visualisation du prix réel vs prédit
    st.subheader("Comparaison : Prix Réel vs Prix Prédit")
    fig_pred, ax_pred = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, ax=ax_pred, alpha=0.6)
    ax_pred.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Ligne de prédiction parfaite')
    ax_pred.set_xlabel("Prix Réel")
    ax_pred.set_ylabel("Prix Prédit")
    ax_pred.set_title("Prix Réel vs Prix Prédit")
    ax_pred.legend()
    st.pyplot(fig_pred)

    # Résidus
    st.subheader("Analyse des Résidus")
    residuals = y_test - y_pred
    fig_res, ax_res = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_pred, y=residuals, ax=ax_res, alpha=0.6)
    ax_res.axhline(y=0, color='r', linestyle='--')
    ax_res.set_xlabel("Prix Prédit")
    ax_res.set_ylabel("Résidus (Prix Réel - Prix Prédit)")
    ax_res.set_title("Résidus vs Prix Prédits")
    st.pyplot(fig_res)
    st.info("Un nuage de points des résidus dispersé aléatoirement autour de zéro indique que le modèle capture bien la relation linéaire. Toute tendance ou motif peut indiquer un problème avec le modèle ou les variables.")

    # --- Section Prédiction Interactive ---
    st.subheader("Prédiction du Prix d’un Téléphone Neuf")

    # Créer un dictionnaire pour stocker les valeurs d'entrée
    input_data = {}
    
    for feature in selected_features:
        # Tenter d'obtenir les min/max du DataFrame pour des valeurs par défaut plus pertinentes
        min_val = df[feature].min()
        max_val = df[feature].max()

        if pd.api.types.is_integer_dtype(df[feature]) or feature in ['charge_rapide', 'nb_coeurs']:
            # Utiliser slider pour les entiers, avec des pas intelligents
            step = 1
            if 'stockage_interne' == feature: step = 32
            elif 'autonomie_batterie' == feature: step = 100
            elif 'nb_coeurs' == feature: step = 2

            input_data[feature] = st.slider(f"{feature.replace('_', ' ').title()}", 
                                            int(min_val) if not np.isnan(min_val) else 0, 
                                            int(max_val) if not np.isnan(max_val) else 100, 
                                            int(min_val + (max_val - min_val)/2) if not np.isnan(min_val) and not np.isnan(max_val) else 50, 
                                            step=step, key=f"pred_slider_{feature}")
        else:
            # Utiliser number_input pour les flottants
            input_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", 
                                                float(min_val) if not np.isnan(min_val) else 0.0, 
                                                float(max_val) if not np.isnan(max_val) else 100.0, 
                                                float(min_val + (max_val - min_val)/2) if not np.isnan(min_val) and not np.isnan(max_val) else 50.0, 
                                                key=f"pred_input_{feature}")

    # Convertir les données d'entrée en DataFrame
    donnees_prediction = pd.DataFrame([input_data])
    
    # Assurer l'ordre des colonnes pour la prédiction
    donnees_prediction = donnees_prediction[selected_features]

    prix_pred = model.predict(donnees_prediction)[0]
    st.success(f"💰 Le prix estimé de ce téléphone est : **{prix_pred:.2f} MAD**")

    # Sauvegarde du modèle
joblib.dump(model, "modele_prix_telephone.pkl")