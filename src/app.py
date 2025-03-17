# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from src.visualization import Visualization
from src.ml_analysis import MLAnalysis
from src.utils import create_interactive_map

# Configuration de la page
st.set_page_config(
    page_title="Analyse des villes françaises",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre de l'application
st.title("Analyse des coûts et qualité de vie dans les villes françaises")
st.markdown("Cette application visualise et analyse les données sur le coût de la vie, la santé et la criminalité dans différentes villes françaises.")

# Fonction pour charger les données
@st.cache_data
def load_data():
    """Charge et prépare les données pour l'application"""
    try:
        # Chargement des données
        dfs = {}
        for data_type in ['cost', 'health', 'crime']:
            try:
                file_path = f'data/load/{data_type}_final.csv'
                if os.path.exists(file_path):
                    dfs[data_type] = pd.read_csv(file_path, index_col='City')
            except Exception as e:
                st.warning(f"Erreur lors du chargement des données {data_type}: {str(e)}")
        
        # Chargement du dataset fusionné s'il existe
        merged_path = 'data/load/merged_dataset.csv'
        if os.path.exists(merged_path):
            merged_df = pd.read_csv(merged_path)
            dfs['merged'] = merged_df
        
        # Chargement des modèles s'ils existent
        models = {}
        models_dir = 'data/models'
        if os.path.exists(models_dir):
            for model_file in os.listdir(models_dir):
                if model_file.endswith('.pkl'):
                    try:
                        model_path = os.path.join(models_dir, model_file)
                        models[model_file] = joblib.load(model_path)
                    except Exception as e:
                        st.warning(f"Erreur lors du chargement du modèle {model_file}: {str(e)}")
        
        return dfs, models
    
    except Exception as e:
        st.error(f"Erreur lors du chargement des données: {str(e)}")
        return {}, {}

# Chargement des données
dfs, models = load_data()

# Barre latérale pour la navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choisir une page",
    ["Vue d'ensemble", "Coût de la vie", "Santé", "Prédictions", "Comparaison des villes"]
)

# Initialisation des objets
viz = Visualization()
ml = MLAnalysis()

# Page: Vue d'ensemble
if page == "Vue d'ensemble":
    st.header("Vue d'ensemble des villes")
    
    # Affichage de la carte si les données sont disponibles
    if 'cost' in dfs:
        st.subheader("Carte des villes analysées")
        
        # Sélection d'une métrique pour la coloration
        metrics = ["Prix moyen d'un appartement", "Salaire mensuel moyen", "Indice de criminalité", "Indice de santé"]
        selected_metric = st.selectbox("Choisir une métrique pour la coloration", metrics)
        
        # Mapping des métriques aux colonnes
        metric_mapping = {
            "Prix moyen d'un appartement": "Apartment (1 bedroom) in City Centre",
            "Salaire mensuel moyen": "Average Monthly Net Salary (After Tax)",
            "Indice de criminalité": "Crime Index_value" if "Crime Index_value" in dfs.get('crime', pd.DataFrame()).columns else None,
            "Indice de santé": "Health Care Index_value" if "Health Care Index_value" in dfs.get('health', pd.DataFrame()).columns else None
        }
        
        # Création du DataFrame pour la carte
        map_df = dfs['cost'].copy()
        
        # Ajout de la métrique sélectionnée depuis le bon DataFrame
        selected_column = metric_mapping[selected_metric]
        if selected_column:
            if selected_metric in ["Prix moyen d'un appartement", "Salaire mensuel moyen"]:
                if selected_column in map_df.columns:
                    map_df = map_df.copy()
            elif selected_metric == "Indice de criminalité" and 'crime' in dfs:
                if selected_column in dfs['crime'].columns:
                    map_df[selected_column] = dfs['crime'][selected_column]
            elif selected_metric == "Indice de santé" and 'health' in dfs:
                if selected_column in dfs['health'].columns:
                    map_df[selected_column] = dfs['health'][selected_column]
        
        # Création de la carte interactive
        if selected_column and selected_column in map_df.columns:
            map_fig = create_interactive_map(
                map_df, 
                map_title=f"Carte des villes françaises - {selected_metric}",
                value_column=selected_column
            )
            st.plotly_chart(map_fig, use_container_width=True)
        else:
            st.warning(f"La métrique {selected_metric} n'est pas disponible dans les données.")
    
    # Statistiques générales
    st.subheader("Statistiques générales")
    
    # Affichage des principales statistiques
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Nombre de villes analysées", len(dfs.get('cost', pd.DataFrame())))
    
    with col2:
        if 'cost' in dfs and 'Average Monthly Net Salary (After Tax)' in dfs['cost'].columns:
            avg_salary = dfs['cost']['Average Monthly Net Salary (After Tax)'].mean()
            st.metric("Salaire moyen (€)", f"{avg_salary:.2f}")
    
    with col3:
        if 'cost' in dfs and 'Apartment (1 bedroom) in City Centre' in dfs['cost'].columns:
            avg_rent = dfs['cost']['Apartment (1 bedroom) in City Centre'].mean()
            st.metric("Loyer moyen (€)", f"{avg_rent:.2f}")

# Page: Coût de la vie
elif page == "Coût de la vie":
    st.header("Analyse du coût de la vie")
    
    if 'cost' in dfs:
        cost_df = dfs['cost']
        
        # Sélection de la catégorie
        categories = {
            "Alimentation": [col for col in cost_df.columns if any(x in col for x in ['Meal', 'McMeal', 'Milk', 'Bread', 'Rice', 'Eggs', 'Cheese', 'Chicken', 'Beef', 'Apples', 'Banana', 'Oranges', 'Tomato', 'Potato', 'Onion', 'Lettuce', 'Water', 'Wine', 'Beer'])],
            "Logement": [col for col in cost_df.columns if any(x in col for x in ['Apartment', 'Square Meter', 'Basic (Electricity', 'Internet', 'Mortgage'])],
            "Transport": [col for col in cost_df.columns if any(x in col for x in ['Ticket', 'Monthly Pass', 'Taxi', 'Gasoline', 'Volkswagen', 'Toyota'])],
            "Loisirs": [col for col in cost_df.columns if any(x in col for x in ['Cinema', 'Fitness', 'Tennis', 'Cappuccino', 'Beer', 'Wine'])],
        }
        
        selected_category = st.selectbox("Choisir une catégorie", list(categories.keys()))
        
        # Sélection des éléments à comparer
        selected_items = st.multiselect(
            "Choisir les éléments à comparer",
            categories[selected_category],
            default=categories[selected_category][:5]
        )
        
        if selected_items:
            # Création du graphique de comparaison
            st.subheader(f"Comparaison des coûts - {selected_category}")
            
            # Création d'un graphique à barres
            fig = px.bar(
                cost_df[selected_items].reset_index(),
                x='City',
                y=selected_items,
                barmode='group',
                title=f"Comparaison des coûts par ville - {selected_category}",
                labels={'value': 'Coût (€)', 'variable': 'Élément'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Affichage du tableau de données
            st.subheader("Données détaillées")
            st.dataframe(cost_df[selected_items])
        
        # Création d'un indice global du coût de la vie
        st.subheader("Indice global du coût de la vie")
        
        # Sélection des poids pour chaque catégorie
        st.write("Ajuster l'importance de chaque catégorie:")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            food_weight = st.slider("Alimentation", 1, 10, 5)
        with col2:
            housing_weight = st.slider("Logement", 1, 10, 6)
        with col3:
            transport_weight = st.slider("Transport", 1, 10, 4)
        with col4:
            leisure_weight = st.slider("Loisirs", 1, 10, 3)
        
        # Calcul de l'indice global
        # Normalisation de chaque catégorie
        normalized_df = pd.DataFrame(index=cost_df.index)
        
        for category, cols in categories.items():
            valid_cols = [col for col in cols if col in cost_df.columns]
            if valid_cols:
                category_values = cost_df[valid_cols].mean(axis=1)
                normalized_df[f"{category}_index"] = (category_values - category_values.min()) / (category_values.max() - category_values.min())
        
        # Application des poids et calcul de l'indice global
        if all(f"{cat}_index" in normalized_df.columns for cat in ["Alimentation", "Logement", "Transport", "Loisirs"]):
            normalized_df['global_index'] = (
                food_weight * normalized_df['Alimentation_index'] +
                housing_weight * normalized_df['Logement_index'] +
                transport_weight * normalized_df['Transport_index'] +
                leisure_weight * normalized_df['Loisirs_index']
            ) / (food_weight + housing_weight + transport_weight + leisure_weight)
            
            # Affichage de l'indice global
            fig = px.bar(
                normalized_df.sort_values('global_index').reset_index(),
                x='City',
                y='global_index',
                title="Indice global du coût de la vie",
                labels={'global_index': 'Indice global', 'City': 'Ville'},
                color='global_index',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Page: Santé
elif page == "Santé":
    st.header("Analyse des indicateurs de santé")
    
    if 'health' in dfs:
        health_df = dfs['health']
        
        # Heatmap des indicateurs de santé
        st.subheader("Comparaison des indicateurs de santé par ville")
        
        # Sélection des colonnes numériques uniquement
        numeric_cols = health_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Sélection des indicateurs à afficher
        selected_indicators = st.multiselect(
            "Choisir les indicateurs à comparer",
            numeric_cols,
            default=[col for col in numeric_cols if '_value' in col][:5]
        )
        
        if selected_indicators:
            # Création du graphique de comparaison
            fig = px.bar(
                health_df[selected_indicators].reset_index(),
                x='City',
                y=selected_indicators,
                barmode='group',
                title="Comparaison des indicateurs de santé par ville",
                labels={'value': 'Score', 'variable': 'Indicateur'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Graphique radar pour une ville spécifique
        st.subheader("Profil de santé par ville")
        
        # Sélection de la ville
        selected_city = st.selectbox("Choisir une ville", health_df.index.tolist())
        
        if selected_city:
            # Sélection des métriques pour le radar
            radar_metrics = [col for col in numeric_cols if '_value' in col]
            
            if radar_metrics:
                # Préparation des données pour le radar
                radar_values = health_df.loc[selected_city, radar_metrics].values.flatten().tolist()
                radar_labels = [col.replace('_value', '') for col in radar_metrics]
                
                # Création du graphique radar
                fig = px.line_polar(
                    r=radar_values,
                    theta=radar_labels,
                    line_close=True,
                    range_r=[0, 100],
                    title=f"Profil de santé de {selected_city}"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# Page: Prédictions
elif page == "Prédictions":
    st.header("Prédictions et analyse par apprentissage automatique")
    
    if len(models) > 0:
        # Affichage des modèles disponibles
        st.subheader("Modèles disponibles")
        
        for model_name, model in models.items():
            st.write(f"- {model_name}")
        
        # Prédiction des prix immobiliers
        st.subheader("Prédiction des prix immobiliers")
        
        # Sélection de la ville de référence
        if 'cost' in dfs:
            reference_city = st.selectbox("Choisir une ville de référence", dfs['cost'].index.tolist())
            
            # Sélection du modèle (si plusieurs disponibles)
            regression_models = [name for name in models.keys() if 'regression' in name.lower()]
            
            if regression_models:
                selected_model = st.selectbox("Choisir un modèle de régression", regression_models)
                
                if selected_model in models:
                    model = models[selected_model]
                    
                    # Interface pour ajuster les paramètres
                    st.write("Ajuster les paramètres pour la prédiction:")
                    
                    # Récupération des paramètres de la ville de référence
                    if reference_city and 'merged' in dfs:
                        city_row = dfs['merged'][dfs['merged']['City'] == reference_city]
                        
                        if not city_row.empty:
                            # Sélection des colonnes numériques pertinentes
                            features = [col for col in city_row.columns if col not in ['City'] and city_row[col].dtype in [np.int64, np.float64, float, int]]
                            
                            # Création des sliders pour chaque paramètre
                            adjusted_params = {}
                            
                            # Organisation en colonnes pour une meilleure interface
                            col1, col2 = st.columns(2)
                            
                            for i, feature in enumerate(features[:10]):  # Limitation à 10 paramètres pour la lisibilité
                                col = col1 if i % 2 == 0 else col2
                                try:
                                    current_value = float(city_row[feature].iloc[0])
                                    min_val = max(0, current_value * 0.5)
                                    max_val = current_value * 1.5
                                    
                                    adjusted_params[feature] = col.slider(
                                        feature,
                                        min_value=float(min_val),
                                        max_value=float(max_val),
                                        value=float(current_value),
                                        format="%.2f"
                                    )
                                except (ValueError, TypeError):
                                    st.warning(f"Impossible d'ajuster le paramètre {feature}")
                            
                            # Bouton pour effectuer la prédiction
                            if st.button("Prédire le prix"):
                                try:
                                    # Création du vecteur de caractéristiques
                                    input_features = np.array([adjusted_params[f] for f in features[:10]]).reshape(1, -1)
                                    
                                    # Prédiction avec le modèle
                                    prediction = model.predict(input_features)
                                    
                                    # Affichage du résultat
                                    st.success(f"Prix prédit: {prediction[0]:.2f} €")
                                except Exception as e:
                                    st.error(f"Erreur lors de la prédiction: {str(e)}")
            else:
                st.warning("Aucun modèle de régression n'est disponible.")
    else:
        st.warning("Aucun modèle disponible. Exécutez d'abord 'main.py' pour générer les modèles.")

# Page: Comparaison des villes
elif page == "Comparaison des villes":
    st.header("Comparaison détaillée entre villes")
    
    if 'cost' in dfs:
        # Sélection des villes à comparer
        available_cities = dfs['cost'].index.tolist()
        
        col1, col2 = st.columns(2)
        
        with col1:
            city1 = st.selectbox("Choisir la première ville", available_cities, index=0)
        
        with col2:
            city2 = st.selectbox("Choisir la deuxième ville", available_cities, index=min(1, len(available_cities)-1))
        
        if city1 != city2:
            # Calcul des différences
            st.subheader(f"Comparaison entre {city1} et {city2}")
            
            # Différences de coût de la vie
            st.markdown("### Coût de la vie")
            
            cost_df = dfs['cost']
            
            # Sélection des catégories à comparer
            categories = {
                "Alimentation": [col for col in cost_df.columns if any(x in col for x in ['Meal', 'McMeal', 'Milk', 'Bread', 'Rice', 'Eggs', 'Cheese', 'Chicken', 'Beef', 'Apples', 'Banana', 'Oranges', 'Tomato', 'Potato', 'Onion', 'Lettuce', 'Water', 'Wine', 'Beer'])],
                "Logement": [col for col in cost_df.columns if any(x in col for x in ['Apartment', 'Square Meter', 'Basic (Electricity', 'Internet', 'Mortgage'])],
                "Transport": [col for col in cost_df.columns if any(x in col for x in ['Ticket', 'Monthly Pass', 'Taxi', 'Gasoline', 'Volkswagen', 'Toyota'])],
                "Loisirs": [col for col in cost_df.columns if any(x in col for x in ['Cinema', 'Fitness', 'Tennis', 'Cappuccino', 'Beer', 'Wine'])],
            }
            
            selected_category = st.selectbox("Choisir une catégorie", list(categories.keys()))
            
            # Comparaison des éléments sélectionnés
            if selected_category in categories:
                selected_items = categories[selected_category]
                
                # Calcul des différences en pourcentage
                if selected_items:
                    comparison_data = []
                    
                    for item in selected_items:
                        if item in cost_df.columns:
                            val1 = cost_df.loc[city1, item]
                            val2 = cost_df.loc[city2, item]
                            
                            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) and val1 > 0:
                                diff_pct = ((val2 - val1) / val1) * 100
                                comparison_data.append({
                                    'Item': item,
                                    f'{city1}': val1,
                                    f'{city2}': val2,
                                    'Différence (%)': diff_pct
                                })
                    
                    if comparison_data:
                        comparison_df = pd.DataFrame(comparison_data)
                        
                        # Création du graphique de comparaison
                        fig = px.bar(
                            comparison_df,
                            x='Item',
                            y='Différence (%)',
                            title=f"Différence de coût entre {city2} et {city1} (en %)",
                            color='Différence (%)',
                            color_continuous_scale=['green', 'white', 'red'],
                            range_color=[-50, 50]
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Affichage du tableau de données
                        st.dataframe(comparison_df.set_index('Item').sort_values('Différence (%)', ascending=False))
            
            # Comparaison des indicateurs de santé si disponibles
            if 'health' in dfs:
                st.markdown("### Indicateurs de santé")
                
                health_df = dfs['health']
                health_metrics = [col for col in health_df.columns if '_value' in col]
                
                if health_metrics:
                    health_comparison = []
                    
                    for metric in health_metrics:
                        if metric in health_df.columns:
                            val1 = health_df.loc[city1, metric] if city1 in health_df.index and metric in health_df.columns else None
                            val2 = health_df.loc[city2, metric] if city2 in health_df.index and metric in health_df.columns else None
                            
                            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                                diff = val2 - val1
                                health_comparison.append({
                                    'Indicateur': metric.replace('_value', ''),
                                    f'{city1}': val1,
                                    f'{city2}': val2,
                                    'Différence': diff
                                })
                    
                    if health_comparison:
                        health_comparison_df = pd.DataFrame(health_comparison)
                        
                        # Création du graphique radar pour comparer les deux villes
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=health_comparison_df[f'{city1}'].tolist(),
                            theta=health_comparison_df['Indicateur'].tolist(),
                            fill='toself',
                            name=city1
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=health_comparison_df[f'{city2}'].tolist(),
                            theta=health_comparison_df['Indicateur'].tolist(),
                            fill='toself',
                            name=city2
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 100]
                                )
                            ),
                            title=f"Comparaison des indicateurs de santé entre {city1} et {city2}"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Veuillez sélectionner deux villes différentes pour la comparaison.")
    else:
        st.warning("Données de coût de la vie non disponibles. Exécutez d'abord 'main.py' pour générer les données.")

# Pied de page
st.markdown("---")
st.markdown("© 2025 | Analyse des coûts et qualité de vie dans les villes françaises")