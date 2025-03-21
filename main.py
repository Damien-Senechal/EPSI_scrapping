# main.py
import os
import pandas as pd
import numpy as np
import warnings
import json
import logging
from src.scraping import Numbeo
from src.data_cleaning import DataCleaner
from src.data_storage import DataStorage
from src.ml_analysis import MLAnalysis
from src.visualization import Visualization

def main(use_example_data=True):
    """
    Fonction principale qui orchestre le pipeline de données complet:
    Extraction -> Nettoyage -> Stockage -> Analyse -> Visualisation
    
    Args:
        use_example_data: Si True, utilise les données d'exemple au lieu du scraping
    """
    # Configuration du logging au lieu de filtrer tous les avertissements
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename='logs/main.log',
        filemode='a'
    )
    logger = logging.getLogger('main')
    
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
    for directory in ['data/extract', 'data/transform', 'data/load', 'data/models', 'data/visualizations', 'logs']:
        os.makedirs(directory, exist_ok=True)
    
    print("Démarrage du pipeline de données...")
    logger.info("Démarrage du pipeline de données")
    
    # =============================================
    # 1. Extraction des données ou chargement des exemples
    # =============================================
    print("\n1. Extraction/Chargement des données...")
    
    if use_example_data:
        print("Utilisation des données d'exemple au lieu du scraping")
        logger.info("Utilisation des données d'exemple")
        
        # Chargement des données d'exemple
        try:
            with open('data/extract/cost_of_living_raw.json', 'r', encoding='utf-8') as f:
                cost_raw_data = json.load(f)
            logger.info("Données d'exemple de coût de la vie chargées")
            
            with open('data/extract/health_raw.json', 'r', encoding='utf-8') as f:
                health_raw_data = json.load(f)
            logger.info("Données d'exemple de santé chargées")
            
            # Pour les données de criminalité, si le fichier n'existe pas, on crée des données minimales
            try:
                with open('data/extract/crime_raw.json', 'r', encoding='utf-8') as f:
                    crime_raw_data = json.load(f)
                logger.info("Données d'exemple de criminalité chargées")
            except FileNotFoundError:
                # Créer des données minimales
                crime_raw_data = []
                for city in cities:
                    crime_raw_data.append({
                        'City': city,
                        'Crime Index': '35.0 Moderate',
                        'Safety Index': '65.0 High'
                    })
                logger.warning("Fichier d'exemple de criminalité non trouvé, création de données minimales")
        except Exception as e:
            logger.error(f"Erreur lors du chargement des données d'exemple: {str(e)}")
            print(f"Erreur lors du chargement des données d'exemple: {str(e)}")
            return
    else:
        # Extraction des données via scraping
        try:
            cost_raw_data = numbeo.extract_cost_of_living(cities)
            health_raw_data = numbeo.extract_health(cities)
            crime_raw_data = numbeo.extract_crime(cities)
        except Exception as e:
            logger.error(f"Erreur lors du scraping: {str(e)}")
            print(f"Erreur lors du scraping: {str(e)}")
            print("Essayez avec use_example_data=True pour utiliser les données d'exemple")
            return
    
    # =============================================
    # 2. Nettoyage des données
    # =============================================
    print("\n2. Nettoyage des données...")
    logger.info("Début du nettoyage des données")
    
    try:
        cost_cleaned_data = cleaner.clean_cost_of_living(cost_raw_data)
        health_cleaned_data = cleaner.clean_health(health_raw_data)
        crime_cleaned_data = cleaner.clean_crime(crime_raw_data)
        logger.info("Nettoyage des données terminé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
        print(f"Erreur lors du nettoyage des données: {str(e)}")
        # Si les données d'exemple nettoyées existent, essayer de les charger
        if use_example_data:
            try:
                print("Tentative de chargement des données nettoyées d'exemple...")
                with open('data/transform/cost_of_living_cleaned.json', 'r', encoding='utf-8') as f:
                    cost_cleaned_data = json.load(f)
                with open('data/transform/health_cleaned.json', 'r', encoding='utf-8') as f:
                    health_cleaned_data = json.load(f)
                try:
                    with open('data/transform/crime_cleaned.json', 'r', encoding='utf-8') as f:
                        crime_cleaned_data = json.load(f)
                except FileNotFoundError:
                    crime_cleaned_data = crime_raw_data  # Utiliser les données brutes si pas de données nettoyées
                logger.info("Utilisation des données nettoyées d'exemple")
            except FileNotFoundError:
                logger.error("Impossible de charger les données nettoyées d'exemple")
                print("Impossible de poursuivre sans données nettoyées")
                return
        else:
            return
    
    # =============================================
    # 3. Stockage des données
    # =============================================
    print("\n3. Stockage des données finales...")
    logger.info("Début du stockage des données")
    
    try:
        dfs = storage.save_final_data(cost_cleaned_data, health_cleaned_data, crime_cleaned_data)
        logger.info("Stockage des données terminé avec succès")
    except Exception as e:
        logger.error(f"Erreur lors du stockage des données: {str(e)}")
        print(f"Erreur lors du stockage des données: {str(e)}")
        # Tenter une approche alternative pour créer les DataFrames
        try:
            print("Tentative de création manuelle des DataFrames...")
            dfs = {}
            dfs['cost'] = pd.DataFrame(cost_cleaned_data).set_index('City')
            dfs['health'] = pd.DataFrame(health_cleaned_data).set_index('City')
            dfs['crime'] = pd.DataFrame(crime_cleaned_data).set_index('City')
            logger.info("Création manuelle des DataFrames réussie")
        except Exception as e:
            logger.error(f"Échec de la création manuelle des DataFrames: {str(e)}")
            print("Impossible de poursuivre sans DataFrames")
            return
    
    # =============================================
    # 4. Préparation des données pour ML
    # =============================================
    print("\n4. Préparation des données pour analyse...")
    logger.info("Début de la préparation des données pour ML")
    
    try:
        # Fusion des datasets ou chargement du dataset fusionné existant
        if use_example_data and os.path.exists('data/load/merged_dataset.csv'):
            print("Chargement du dataset fusionné d'exemple...")
            merged_df = pd.read_csv('data/load/merged_dataset.csv')
            logger.info("Dataset fusionné d'exemple chargé")
        else:
            all_dfs = list(dfs.values())
            merged_df = storage.merge_datasets(all_dfs)
            logger.info("Fusion des datasets réussie")
        
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
        logger.info("Préparation des données terminée et sauvegardée")
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des données: {str(e)}")
        print(f"Erreur lors de la préparation des données: {str(e)}")
        return
    
    # =============================================
    # 5. Visualisations
    # =============================================
    print("\n5. Création des visualisations de base...")
    logger.info("Début de la création des visualisations")
    
    try:
        # Comparaison des coûts pour une sélection d'éléments
        cost_items = [col for col in dfs['cost'].columns if 'Restaurant' in col or 'Cappuccino' in col or 'Monthly Pass' in col or 'Cinema' in col]
        if cost_items:
            cost_items = cost_items[:4]  # Limiter à 4 items
            viz.plot_cost_comparison(dfs['cost'], cost_items)
            logger.info("Visualisation de comparaison des coûts créée")
        
        # Heatmap de santé - vérifier que les colonnes numériques existent
        health_numeric_cols = dfs['health'].select_dtypes(include=[np.number]).columns
        if len(health_numeric_cols) > 0:
            viz.plot_health_heatmap(dfs['health'], "Métriques de santé par ville")
            logger.info("Heatmap de santé créée")
        
        # Graphiques radar par ville - vérifier les métriques disponibles
        health_metrics = [col for col in dfs['health'].columns if '_value' in col]
        if health_metrics:
            metrics = health_metrics[:4]  # Prendre les 4 premières métriques disponibles
            for city in dfs['health'].index:
                if all(metric in dfs['health'].columns for metric in metrics):
                    viz.plot_city_radar(dfs['health'], city, metrics)
                    logger.info(f"Graphique radar pour {city} créé")
        
        # Analyse PCA
        viz.plot_pca_analysis(numeric_df, n_components=2, title="ACP des données de ville")
        logger.info("Analyse PCA créée")
    except Exception as e:
        logger.error(f"Erreur lors de la création des visualisations: {str(e)}")
        print(f"Erreur lors de la création des visualisations: {str(e)}")
        # Continuer malgré l'erreur
    
    # =============================================
    # 6. Machine Learning - Régression
    # =============================================
    print("\n6. Analyse par régression...")
    logger.info("Début de l'analyse par régression")
    
    try:
        # Vérifier qu'il y a suffisamment de données pour la régression
        if len(numeric_df) >= 5:  # Au moins 5 villes pour faire de la régression
            # Chercher une cible appropriée pour la régression
            potential_targets = [col for col in numeric_df.columns if 'Apartment' in col or 'Price' in col]
            target_column = potential_targets[0] if potential_targets else None
            
            if target_column:
                # Préparation des données pour ML avec gestion d'erreur
                try:
                    X_train, X_test, y_train, y_test = ml.prepare_data(numeric_df, target_column)
                    
                    # Régression linéaire
                    linear_results = ml.regression_analysis(X_train, X_test, y_train, y_test, model_type='linear')
                    print(f"R² de régression linéaire: {linear_results.get('r2', 'N/A')}")
                    
                    # Régression par forêt aléatoire
                    rf_results = ml.regression_analysis(X_train, X_test, y_train, y_test, model_type='random_forest')
                    print(f"R² de régression par forêt aléatoire: {rf_results.get('r2', 'N/A')}")
                    
                    # Graphique des résultats de régression
                    if 'predictions' in rf_results:
                        viz.plot_regression_results(y_test, rf_results['predictions'], 
                                                "Prédiction des prix immobiliers - Valeurs prédites vs. réelles")
                    
                    # Graphique d'importance des features
                    if 'feature_importance' in rf_results:
                        viz.plot_feature_importance(rf_results['feature_importance'], 
                                                "Prédiction des prix immobiliers - Importance des caractéristiques")
                    
                    logger.info("Analyse par régression terminée avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse par régression: {str(e)}")
                    print(f"Erreur lors de l'analyse par régression: {str(e)}")
            else:
                logger.warning("Aucune cible appropriée trouvée pour la régression")
                print("Aucune cible appropriée trouvée pour la régression")
        else:
            logger.warning("Pas assez de données pour effectuer une régression")
            print("Pas assez de données pour effectuer une régression")
    except Exception as e:
        logger.error(f"Erreur non gérée lors de l'analyse par régression: {str(e)}")
        print(f"Erreur non gérée lors de l'analyse par régression: {str(e)}")
    
    # =============================================
    # 7. Machine Learning - Classification
    # =============================================
    print("\n7. Analyse par classification...")
    logger.info("Début de l'analyse par classification")
    
    try:
        # Vérifier qu'il y a suffisamment de données pour la classification
        if len(numeric_df) >= 5:
            # Chercher une cible appropriée pour la classification
            potential_targets = [col for col in numeric_df.columns if 'Salary' in col or 'Income' in col]
            salary_col = potential_targets[0] if potential_targets else None
            
            if salary_col:
                # Création de catégories basées sur le salaire
                try:
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
                    
                    print(f"Précision de classification: {rf_class_results.get('accuracy', 'N/A')}")
                    print(f"Score F1 de classification: {rf_class_results.get('f1', 'N/A')}")
                    
                    if 'classification_report' in rf_class_results:
                        print("\nRapport de classification:")
                        print(rf_class_results['classification_report'])
                    
                    # Graphique d'importance des features pour la classification
                    if 'feature_importance' in rf_class_results:
                        viz.plot_feature_importance(
                            rf_class_results['feature_importance'], 
                            "Classification des catégories de salaire - Importance des caractéristiques"
                        )
                    
                    logger.info("Analyse par classification terminée avec succès")
                except Exception as e:
                    logger.error(f"Erreur lors de l'analyse par classification: {str(e)}")
                    print(f"Erreur lors de l'analyse par classification: {str(e)}")
            else:
                logger.warning("Aucune cible appropriée trouvée pour la classification")
                print("Aucune cible appropriée trouvée pour la classification")
        else:
            logger.warning("Pas assez de données pour effectuer une classification")
            print("Pas assez de données pour effectuer une classification")
    except Exception as e:
        logger.error(f"Erreur non gérée lors de l'analyse par classification: {str(e)}")
        print(f"Erreur non gérée lors de l'analyse par classification: {str(e)}")
    
    print("\nPipeline de données terminé avec succès.")
    logger.info("Pipeline de données terminé avec succès")

if __name__ == "__main__":
    # Utiliser les données d'exemple par défaut pour éviter les problèmes de scraping
    # Mettre à False pour tenter le scraping réel
    main(use_example_data=True)