# src/data_cleaning.py
import pandas as pd
import numpy as np
import json
import os
import re

class DataCleaner:
    """
    Classe pour le nettoyage et la transformation des données brutes.
    Effectue des conversions de types, normalisations et gère les valeurs manquantes.
    """
    
    def __init__(self):
        """Initialise l'objet de nettoyage de données"""
        # Création des répertoires de stockage
        os.makedirs('data/transform', exist_ok=True)
    
    def clean_cost_of_living(self, raw_data):
        """
        Nettoyage des données du coût de la vie et conversion des types
        
        Args:
            raw_data: données brutes du coût de la vie (liste de dictionnaires)
            
        Returns:
            cleaned_data: données nettoyées (liste de dictionnaires)
        """
        cleaned_data = []
        for city_data in raw_data:
            cleaned_city_data = {'City': city_data['City']}
            for key, value in city_data.items():
                if key != 'City':
                    # Nettoyer et convertir en type approprié
                    cleaned_city_data[key] = self._convert_to_numeric(self._clean_price(value))
            cleaned_data.append(cleaned_city_data)

        # Sauvegarde des données nettoyées
        self._save_cleaned_data(cleaned_data, 'cost_of_living')
        
        return cleaned_data
    
    def clean_health(self, raw_data):
        """
        Nettoyage des données de santé et séparation des valeurs numériques
        
        Args:
            raw_data: données brutes de santé (liste de dictionnaires)
            
        Returns:
            cleaned_data: données nettoyées (liste de dictionnaires)
        """
        cleaned_data = []
        for city_data in raw_data:
            cleaned_city_data = {'City': city_data['City']}
            for key, value in city_data.items():
                if key != 'City':
                    try:
                        # Remplacer \n par un espace et nettoyer
                        value = value.replace('\n', ' ').strip()
                        
                        # Séparer la partie numérique et textuelle
                        if value.split(' ')[0].replace('.', '').isdigit():
                            number = float(value.split(' ')[0])
                            rating = ' '.join(value.split(' ')[1:])  # Partie textuelle (High, Very High, etc.)
                            
                            # Créer deux colonnes distinctes pour la valeur numérique et le rating
                            cleaned_city_data[f"{key}_value"] = number
                            if rating:  # Si le rating existe
                                cleaned_city_data[f"{key}_rating"] = rating
                        else:
                            cleaned_city_data[key] = value
                    except (ValueError, AttributeError):
                        cleaned_city_data[key] = value

            cleaned_data.append(cleaned_city_data)

        # Sauvegarde des données nettoyées
        self._save_cleaned_data(cleaned_data, 'health')
        
        return cleaned_data
    
    def clean_crime(self, raw_data):
        """
        Nettoyage des données de criminalité et conversion des types
        
        Args:
            raw_data: données brutes de criminalité (liste de dictionnaires)
            
        Returns:
            cleaned_data: données nettoyées (liste de dictionnaires)
        """
        cleaned_data = []
        for city_data in raw_data:
            cleaned_city_data = {'City': city_data['City']}
            for key, value in city_data.items():
                if key != 'City':
                    try:
                        # Nettoyer la valeur
                        value = value.replace('\n', ' ').strip()
                        
                        # Séparer la partie numérique et textuelle (si applicable)
                        if value.split(' ')[0].replace('.', '').isdigit():
                            number = float(value.split(' ')[0])
                            level = ' '.join(value.split(' ')[1:])
                            
                            cleaned_city_data[f"{key}_value"] = number
                            if level:
                                cleaned_city_data[f"{key}_level"] = level
                        else:
                            cleaned_city_data[key] = value
                    except (ValueError, AttributeError):
                        cleaned_city_data[key] = value
            
            cleaned_data.append(cleaned_city_data)
        
        # Sauvegarde des données nettoyées
        self._save_cleaned_data(cleaned_data, 'crime')
        
        return cleaned_data
    
    def _clean_price(self, price_text):
        """
        Nettoyage des valeurs de prix - gestion des séparateurs de milliers et décimaux
        """
        try:
            # Supprimer les espaces et le symbole €
            price = str(price_text).replace(' ', '').replace('€', '').strip()
            
            # Traiter au cas où il y a plusieurs points (séparateurs de milliers)
            if price.count('.') > 1:
                # Supprimer tous les points (séparateurs de milliers)
                price = price.replace('.', '')
            
            # Remplacer la virgule par un point pour les décimales
            price = price.replace(',', '.')
            
            # Vérifier si c'est un nombre valide
            if price.replace('.', '', 1).isdigit():  # un seul point autorisé
                # Convertir en nombre à virgule flottante
                return float(price)
            
            return price_text.strip().replace(' €', '')
        except Exception as e:
            print(f"Erreur lors du nettoyage du prix: {e}")
            return price_text.strip().replace(' €', '')
    
    def _convert_to_numeric(self, value_str):
        """
        Convertit une chaîne en type numérique approprié (float ou int)
        
        Args:
            value_str: chaîne à convertir
            
        Returns:
            numeric_value: valeur numérique convertie ou chaîne d'origine si impossible
        """
        try:
            # Tenter de convertir en float
            value_float = float(value_str)
            
            # Si c'est un entier (pas de partie décimale), convertir en int
            if value_float.is_integer():
                return int(value_float)
            else:
                return value_float
        except (ValueError, TypeError):
            # Si la conversion échoue, retourner la valeur originale
            return value_str
    
    def _save_cleaned_data(self, data, data_type):
        """
        Sauvegarde des données nettoyées dans des fichiers JSON et CSV
        
        Args:
            data: liste de dictionnaires avec les données nettoyées
            data_type: type de données (cost_of_living, health, crime)
        """
        # Sauvegarde en JSON
        with open(f'data/transform/{data_type}_cleaned.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Sauvegarde en CSV
        df = pd.DataFrame(data)
        df.to_csv(f'data/transform/{data_type}_cleaned.csv', index=False)
    
    def normalize_data(self, df, columns=None, method='minmax'):
        """
        Normalise les colonnes numériques d'un DataFrame
        
        Args:
            df: DataFrame à normaliser
            columns: liste des colonnes à normaliser (None = toutes les colonnes numériques)
            method: méthode de normalisation ('minmax', 'zscore')
            
        Returns:
            normalized_df: DataFrame normalisé
        """
        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        
        # Copie pour ne pas modifier l'original
        normalized_df = df.copy()
        
        # Sélection des colonnes à normaliser
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        # Application de la normalisation
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'zscore':
            scaler = StandardScaler()
        else:
            raise ValueError("Méthode de normalisation invalide. Utilisez 'minmax' ou 'zscore'.")
        
        normalized_df[columns] = scaler.fit_transform(normalized_df[columns])
        
        return normalized_df
    
    def one_hot_encode(self, df, columns):
        """
        Applique le one-hot encoding aux colonnes catégorielles spécifiées
        
        Args:
            df: DataFrame avec les données
            columns: liste des colonnes à encoder
            
        Returns:
            encoded_df: DataFrame avec les colonnes encodées
        """
        encoded_df = pd.get_dummies(df, columns=columns, drop_first=True)
        return encoded_df
    
    def manage_missing_values(self, df, strategy='mean'):
        """
        Gère les valeurs manquantes dans un DataFrame
        
        Args:
            df: DataFrame avec les données
            strategy: stratégie pour remplir les valeurs manquantes ('mean', 'median', 'mode', 'drop')
            
        Returns:
            df_processed: DataFrame avec les valeurs manquantes traitées
        """
        if strategy == 'drop':
            return df.dropna()
        
        df_processed = df.copy()
        
        # Traitement pour les colonnes numériques
        numeric_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if strategy == 'mean':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mean())
            elif strategy == 'median':
                df_processed[col] = df_processed[col].fillna(df_processed[col].median())
            elif strategy == 'mode':
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        # Traitement pour les colonnes catégorielles
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            # Pour les colonnes catégorielles, utiliser le mode
            if df_processed[col].isnull().any():
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
        
        return df_processed