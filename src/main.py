# main.py
import os
import pandas as pd
import numpy as np
import warnings
from src.scraping import Numbeo
from src.data_cleaning import DataCleaner
from src.data_storage import DataStorage
from src.ml_analysis import MLAnalysis
from src.visualization import Visualization

def main():
    """
    Fonction principale qui orchestre le pipeline de données complet:
    Extraction -> Nettoyage -> Stockage -> Analyse -> Visualisation
    """
    # Ignorer les avertissements pour une sortie plus propre
    warnings.filterwarnings('ignore')
    
    # Configuration
    cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Nantes", "Nice"]
    use_database = False  # Mettre à True pour utiliser une base de données SQLite
    
    # Initialisation des composants
    numbeo = Numbeo()
    cleaner = DataCleaner()
    storage = DataStorage(use_db=use_database)
    ml = MLAnalysis()
    viz = Visualization()
    
    # Vérification de l'existence des répertoires et création si nécessaire
    for directory in ['data/extract', 'data/transform', 'data/load', 'data/models', 'data/visualizations']:
        os.makedirs(directory, exist_ok=True)
    
    print("Démarrage du pipeline de données...")
    
    # =============================================
    # 1. Extraction des données
    # =============================================
    print("\n1. Extraction des données...")
    cost_raw_data = numbeo.extract_cost_of_living(cities)
    health_raw_data = numbeo.extract_health(cities)
    crime_raw_data = numbeo.extract_crime(cities)
    
    # =============================================
    # 2. Nettoyage des données
    # =============================================
    print("\n2. Nettoyage des données...")
    cost_cleaned_data = cleaner.clean_cost_of_living(cost_raw_data)
    health_cleaned_data = cleaner.clean_health(health_raw_data)
    crime_cleaned_data = cleaner.clean_crime(crime_raw_data)
    
    # =============================================
    # 3. Stockage des données
    # =============================================
    print("\n3. Stockage des données finales...")
    dfs = storage.save_final_data(cost_cleaned_data, health_cleaned_data, crime_cleaned_data)
    
    # =============================================
    # 4. Préparation des données pour ML
    # =============================================
    print("\n4. Préparation des données pour analyse...")
    # Fusion des datasets
    all_dfs = list(dfs.values())
    merged_df = storage.merge_datasets(all_dfs)
    
    # Remplacement des valeurs problématiques par NaN (comme '?' dans les données brutes)
    merged_df = merged_df.replace('?', np.nan)
    
    # Suppression des colonnes avec trop de valeurs manquantes
    merged_df = merged_df.dropna(axis=1, thresh=len(merged_df)*0.8)
    
    # Remplissage des valeurs NaN restantes avec la moyenne de la colonne pour les colonnes numériques
    for col in merged_df.select_dtypes(include=[np.number]).columns:
        merged_df[col] = merged_df[col].fillna(merged_df[col].mean())
    
    # Conservation uniquement des colonnes numériques pour l'analyse ML initiale
    numeric_df = merged_df.select_dtypes(include=[np.number])
    
    # Sauvegarde des données préparées
    numeric_df.to_csv('data/load/prepared_data.csv')
    
    # =============================================
    # 5. Visualisations
    # =============================================
    print("\n5. Création des visualisations de base...")
    
    # Comparaison des coûts pour une sélection d'éléments
    cost_items = ['Meal, Inexpensive Restaurant', 'Cappuccino (regular)', 
                  'Monthly Pass (Regular Price)', 'Cinema, International Release, 1 Seat']
    viz.plot_cost_comparison(dfs['cost'], cost_items)
    
    # Heatmap de santé
    viz.plot_health_heatmap(dfs['health'], "Métriques de santé par ville")
    
    # Graphiques radar par ville
    for city in cities:
        metrics = ['Skill and competency of medical staff_value', 
                   'Equipment for modern diagnosis and treatment_value',
                   'Satisfaction with cost to you_value',
                   'Convenience of location for you_value']
        if all(metric in dfs['health'].columns for metric in metrics):
            viz.plot_city_radar(dfs['health'], city, metrics)
    
    # Analyse PCA
    viz.plot_pca_analysis(numeric_df, n_components=2, title="ACP des données de ville")
    
    # =============================================
    # 6. Machine Learning - Régression
    # =============================================
    print("\n6. Analyse par régression...")
    
    # Choix d'une cible pour la régression (ex: prix du logement)
    if 'Apartment (1 bedroom) in City Centre' in numeric_df.columns:
        target_column = 'Apartment (1 bedroom) in City Centre'
        
        # Préparation des données pour ML
        X_train, X_test, y_train, y_test = ml.prepare_data(numeric_df, target_column)
        
        # Régression linéaire
        linear_results = ml.regression_analysis(X_train, X_test, y_train, y_test, model_type='linear')
        print(f"R² de régression linéaire: {linear_results['r2']:.4f}")
        
        # Régression par forêt aléatoire
        rf_results = ml.regression_analysis(X_train, X_test, y_train, y_test, model_type='random_forest')
        print(f"R² de régression par forêt aléatoire: {rf_results['r2']:.4f}")
        
        # Graphique des résultats de régression
        viz.plot_regression_results(y_test, rf_results['predictions'], 
                                   "Prédiction des prix immobiliers - Valeurs prédites vs. réelles")
        
        # Graphique d'importance des features
        if 'feature_importance' in rf_results:
            viz.plot_feature_importance(rf_results['feature_importance'], 
                                       "Prédiction des prix immobiliers - Importance des caractéristiques")
    
    # =============================================
    # 7. Machine Learning - Classification
    # =============================================
    print("\n7. Analyse par classification...")
    
    # Création d'une cible de classification basée sur le coût de la vie
    if 'Average Monthly Net Salary (After Tax)' in numeric_df.columns:
        # Création de catégories basées sur le salaire
        salary_col = 'Average Monthly Net Salary (After Tax)'
        numeric_df_with_target = ml.create_classification_target(
            numeric_df, 
            salary_col, 
            n_categories=3, 
            labels=['Bas', 'Moyen', 'Élevé']
        )
        
        # Préparation des données pour la classification
        X_train, X_test, y_train, y_test = ml.prepare_data(
            numeric_df_with_target.drop(salary_col, axis=1), 
            f'{salary_col}_categorie'
        )
        
        # Classification par forêt aléatoire
        rf_class_results = ml.classification_analysis(
            X_train, X_test, y_train, y_test, 
            model_type='random_forest'
        )
        
        print(f"Précision de classification: {rf_class_results['accuracy']:.4f}")
        print(f"Score F1 de classification: {rf_class_results['f1']:.4f}")
        print("\nRapport de classification:")
        print(rf_class_results['classification_report'])
        
        # Graphique d'importance des features pour la classification
        if 'feature_importance' in rf_class_results:
            viz.plot_feature_importance(
                rf_class_results['feature_importance'], 
                "Classification des catégories de salaire - Importance des caractéristiques"
            )
    
    print("\nPipeline de données terminé avec succès.")

if __name__ == "__main__":
    main()