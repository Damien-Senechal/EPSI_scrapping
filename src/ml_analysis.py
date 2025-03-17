# src/ml_analysis.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class MLAnalysis:
    """
    Classe pour l'analyse par apprentissage automatique des données de villes.
    Implémente des modèles de régression et de classification.
    """
    
    def __init__(self):
        """Initialise l'objet d'analyse ML"""
        # Création du répertoire pour sauvegarder les modèles
        os.makedirs('data/models', exist_ok=True)
        self.feature_names = []
        self.label_encoder = None
    
    def prepare_data(self, df, target_column, test_size=0.2, random_state=42):
        """
        Prépare les données pour l'analyse ML en les divisant en features et target
        
        Args:
            df: DataFrame contenant toutes les données
            target_column: colonne cible à prédire
            test_size: proportion des données pour le test (défaut: 0.2)
            random_state: graine aléatoire pour reproductibilité
            
        Returns:
            X_train, X_test, y_train, y_test: données divisées pour l'entrainement et le test
        """
        # Gestion de la cible catégorielle si nécessaire
        if df[target_column].dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(df[target_column])
            self.label_encoder = le
        else:
            y = df[target_column].values
        
        # Les features sont toutes les colonnes sauf la cible
        X = df.drop(target_column, axis=1)
        
        # Garder la trace des noms des features
        self.feature_names = X.columns.tolist()
        
        # Standardisation des features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scaler = scaler
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test
    
    def regression_analysis(self, X_train, X_test, y_train, y_test, model_type='linear'):
        """
        Effectue une analyse de régression avec le modèle spécifié
        
        Args:
            X_train, X_test, y_train, y_test: données divisées
            model_type: type de modèle ('linear', 'random_forest', 'svr')
            
        Returns:
            results: dictionnaire contenant le modèle et les métriques d'évaluation
        """
        # Création du modèle
        if model_type == 'linear':
            model = LinearRegression()
        elif model_type == 'random_forest':
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        elif model_type == 'svr':
            model = SVR()
        else:
            raise ValueError("Type de modèle invalide. Choisir parmi: 'linear', 'random_forest', 'svr'")
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        # Création du dictionnaire de résultats
        results = {
            'model': model,
            'predictions': y_pred,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores)
        }
        
        # Importance des features pour Random Forest
        if model_type == 'random_forest':
            results['feature_importance'] = dict(zip(self.feature_names, model.feature_importances_))
        
        # Sauvegarde du modèle
        joblib.dump(model, f'data/models/regression_{model_type}.pkl')
        
        return results
    
    def classification_analysis(self, X_train, X_test, y_train, y_test, model_type='logistic'):
        """
        Effectue une analyse de classification avec le modèle spécifié
        
        Args:
            X_train, X_test, y_train, y_test: données divisées
            model_type: type de modèle ('logistic', 'random_forest', 'svc')
            
        Returns:
            results: dictionnaire contenant le modèle et les métriques d'évaluation
        """
        # Création du modèle
        if model_type == 'logistic':
            model = LogisticRegression(max_iter=1000, random_state=42)
        elif model_type == 'random_forest':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'svc':
            model = SVC(probability=True, random_state=42)
        else:
            raise ValueError("Type de modèle invalide. Choisir parmi: 'logistic', 'random_forest', 'svc'")
        
        # Entraînement du modèle
        model.fit(X_train, y_train)
        
        # Prédictions
        y_pred = model.predict(X_test)
        
        # Évaluation du modèle
        accuracy = accuracy_score(y_test, y_pred)
        try:
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
        except:
            # Gestion du cas de classification binaire
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
        
        # Rapport de classification
        report = classification_report(y_test, y_pred)
        
        # Validation croisée
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Création du dictionnaire de résultats
        results = {
            'model': model,
            'predictions': y_pred,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'cv_scores': cv_scores,
            'cv_mean': np.mean(cv_scores),
            'classification_report': report
        }
        
        # Importance des features pour Random Forest
        if model_type == 'random_forest':
            results['feature_importance'] = dict(zip(self.feature_names, model.feature_importances_))
        
        # Sauvegarde du modèle
        joblib.dump(model, f'data/models/classification_{model_type}.pkl')
        
        return results
    
    def optimize_model(self, X_train, y_train, model_type, param_grid):
        """
        Optimise les hyperparamètres d'un modèle avec GridSearchCV
        
        Args:
            X_train, y_train: données d'entraînement
            model_type: type de modèle
            param_grid: grille de paramètres à tester
            
        Returns:
            results: dictionnaire contenant le meilleur modèle et les résultats
        """
        # Configuration du modèle
        if model_type == 'linear':
            model = LinearRegression()
            scoring = 'r2'
        elif model_type == 'random_forest_regressor':
            model = RandomForestRegressor(random_state=42)
            scoring = 'r2'
        elif model_type == 'svr':
            model = SVR()
            scoring = 'r2'
        elif model_type == 'logistic':
            model = LogisticRegression(random_state=42)
            scoring = 'accuracy'
        elif model_type == 'random_forest_classifier':
            model = RandomForestClassifier(random_state=42)
            scoring = 'accuracy'
        elif model_type == 'svc':
            model = SVC(random_state=42)
            scoring = 'accuracy'
        else:
            raise ValueError("Type de modèle invalide")
        
        # Configuration de GridSearchCV
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring=scoring, verbose=1, n_jobs=-1
        )
        
        # Exécution de la recherche par grille
        grid_search.fit(X_train, y_train)
        
        # Obtention des meilleurs paramètres et score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Création du dictionnaire de résultats
        results = {
            'best_model': grid_search.best_estimator_,
            'best_params': best_params,
            'best_score': best_score,
            'all_results': grid_search.cv_results_
        }
        
        # Sauvegarde du meilleur modèle
        joblib.dump(grid_search.best_estimator_, f'data/models/optimized_{model_type}.pkl')
        
        return results
    
    def create_classification_target(self, df, column, n_categories=3, labels=None):
        """
        Crée une variable cible catégorielle à partir d'une colonne numérique
        
        Args:
            df: DataFrame
            column: colonne à transformer
            n_categories: nombre de catégories à créer
            labels: noms des catégories (optionnel)
            
        Returns:
            df_with_target: DataFrame avec la nouvelle colonne cible
        """
        df_new = df.copy()
        
        if labels is None:
            if n_categories == 3:
                labels = ['Bas', 'Moyen', 'Élevé']
            elif n_categories == 2:
                labels = ['Bas', 'Élevé']
            else:
                labels = [f'Catégorie {i+1}' for i in range(n_categories)]
        
        # Création de la catégorie basée sur les quantiles
        df_new[f'{column}_categorie'] = pd.qcut(
            df_new[column], 
            q=n_categories, 
            labels=labels
        )
        
        return df_new
    
    def predict_new_data(self, model_path, new_data):
        """
        Charge un modèle sauvegardé et fait des prédictions sur de nouvelles données
        
        Args:
            model_path: chemin vers le modèle sauvegardé
            new_data: nouvelles données pour prédiction
            
        Returns:
            predictions: prédictions du modèle
        """
        # Chargement du modèle
        model = joblib.load(model_path)
        
        # Application du scaler si disponible
        if hasattr(self, 'scaler'):
            new_data_scaled = self.scaler.transform(new_data)
        else:
            new_data_scaled = new_data
        
        # Prédictions
        predictions = model.predict(new_data_scaled)
        
        # Pour la classification, reconvertir en labels originaux si nécessaire
        if hasattr(self, 'label_encoder') and hasattr(model, 'predict_proba'):
            predictions = self.label_encoder.inverse_transform(predictions)
        
        return predictions