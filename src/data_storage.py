# src/data_storage.py
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime

class DataStorage:
    """
    Classe pour le stockage et la gestion des données.
    Supporte la sauvegarde dans des fichiers et optionnellement dans une base de données SQLite.
    """
    
    def __init__(self, use_db=False, db_path='data/city_data.db'):
        """
        Initialise le stockage de données avec l'option d'utiliser une base de données SQLite
        
        Args:
            use_db: booléen indiquant s'il faut utiliser une base de données
            db_path: chemin vers le fichier de base de données
        """
        self.use_db = use_db
        self.db_path = db_path
        
        # Création des répertoires
        os.makedirs('data/load', exist_ok=True)
        
        # Initialisation de la base de données si nécessaire
        if use_db:
            self._init_database()
    
    def _init_database(self):
        """Initialise la base de données SQLite avec les tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Création des tables pour chaque catégorie de données
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS cost_of_living (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            data_json TEXT,
            created_at TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS health_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            data_json TEXT,
            created_at TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS crime_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT,
            data_json TEXT,
            created_at TIMESTAMP
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_final_data(self, cost_data, health_data, crime_data=None):
        """
        Sauvegarde les données finales dans des fichiers et/ou la base de données
        
        Args:
            cost_data: données de coût de la vie
            health_data: données de santé
            crime_data: données de criminalité (optionnel)
            
        Returns:
            dfs: dictionnaire de DataFrames pour analyse ultérieure
        """
        # Création des DataFrames
        cost_df = pd.DataFrame(cost_data).set_index('City')
        health_df = pd.DataFrame(health_data).set_index('City')
        
        # Sauvegarde en CSV
        cost_df.to_csv('data/load/cost_of_living_final.csv')
        health_df.to_csv('data/load/health_final.csv')
        
        # Sauvegarde en JSON
        with open('data/load/cost_of_living_final.json', 'w', encoding='utf-8') as f:
            json.dump(cost_data, f, ensure_ascii=False, indent=4)
        
        with open('data/load/health_final.json', 'w', encoding='utf-8') as f:
            json.dump(health_data, f, ensure_ascii=False, indent=4)
        
        # Sauvegarde des données de criminalité si disponibles
        if crime_data:
            crime_df = pd.DataFrame(crime_data).set_index('City')
            crime_df.to_csv('data/load/crime_final.csv')
            with open('data/load/crime_final.json', 'w', encoding='utf-8') as f:
                json.dump(crime_data, f, ensure_ascii=False, indent=4)
        
        # Sauvegarde dans la base de données si activée
        if self.use_db:
            self._save_to_database(cost_data, 'cost_of_living')
            self._save_to_database(health_data, 'health_data')
            if crime_data:
                self._save_to_database(crime_data, 'crime_data')
        
        # Retour des dataframes pour analyse ultérieure
        dfs = {
            'cost': cost_df,
            'health': health_df
        }
        if crime_data:
            dfs['crime'] = pd.DataFrame(crime_data).set_index('City')
        
        return dfs
    
    def _save_to_database(self, data_list, table_name):
        """
        Sauvegarde une liste de données dans la base de données SQLite
        
        Args:
            data_list: liste de dictionnaires de données
            table_name: nom de la table dans la base de données
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for item in data_list:
            city = item['City']
            # Conversion en chaîne JSON pour le stockage
            data_json = json.dumps(item, ensure_ascii=False)
            
            cursor.execute(f'''
            INSERT INTO {table_name} (city, data_json, created_at)
            VALUES (?, ?, ?)
            ''', (city, data_json, now))
        
        conn.commit()
        conn.close()
    
    def load_data(self, data_type='all'):
        """
        Charge les données depuis les fichiers ou la base de données
        
        Args:
            data_type: type de données à charger ('all', 'cost', 'health', 'crime')
            
        Returns:
            results: dictionnaire des DataFrames chargés
        """
        results = {}
        
        if data_type in ['all', 'cost']:
            try:
                cost_df = pd.read_csv('data/load/cost_of_living_final.csv', index_col='City')
                results['cost'] = cost_df
            except FileNotFoundError:
                print("Fichier de données sur le coût de la vie non trouvé")
        
        if data_type in ['all', 'health']:
            try:
                health_df = pd.read_csv('data/load/health_final.csv', index_col='City')
                results['health'] = health_df
            except FileNotFoundError:
                print("Fichier de données de santé non trouvé")
        
        if data_type in ['all', 'crime']:
            try:
                crime_df = pd.read_csv('data/load/crime_final.csv', index_col='City')
                results['crime'] = crime_df
            except FileNotFoundError:
                print("Fichier de données de criminalité non trouvé")
        
        return results
    
    def merge_datasets(self, dataframes, on_column='City'):
        """
        Fusionne plusieurs dataframes en un seul dataset d'analyse
        
        Args:
            dataframes: liste de DataFrames à fusionner
            on_column: colonne sur laquelle effectuer la fusion
            
        Returns:
            merged_df: DataFrame résultant de la fusion
        """
        if not dataframes:
            raise ValueError("Aucun dataframe fourni pour la fusion")
        
        # Commencer avec le premier dataframe
        merged_df = dataframes[0].reset_index()
        
        # Fusion avec les autres dataframes
        for i in range(1, len(dataframes)):
            df = dataframes[i].reset_index()
            # Gestion des noms de colonnes qui se chevauchent avec ajout de suffixes
            overlapping = set(merged_df.columns) & set(df.columns) - {on_column}
            if overlapping:
                merged_df = pd.merge(merged_df, df, on=on_column, suffixes=('', f'_{i}'))
            else:
                merged_df = pd.merge(merged_df, df, on=on_column)
        
        # Sauvegarde du dataset fusionné
        merged_df.to_csv('data/load/merged_dataset.csv', index=False)
        
        return merged_df