# src/scraping.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json
import os

class Numbeo:
    """
    Classe pour l'extraction des données depuis Numbeo.com
    Permet de récupérer des données sur le coût de la vie, la santé et la criminalité dans différentes villes.
    """
    
    def __init__(self):
        """Initialise la classe de scraping avec les headers HTTP nécessaires"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        # Création des répertoires de stockage
        os.makedirs('data/extract', exist_ok=True)
    
    def extract_cost_of_living(self, cities):
        """
        Extraction des données du coût de la vie pour plusieurs villes
        
        Args:
            cities: liste de noms de villes
            
        Returns:
            raw_data: liste de dictionnaires contenant les données brutes
        """
        raw_data = []
        for city in cities:
            try:
                print(f"Extraction des données de coût de la vie pour {city}...")
                url = f"https://www.numbeo.com/cost-of-living/in/{city}?displayCurrency=EUR"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                tables = soup.find_all('table', class_='data_wide_table')
                
                data = {'City': city}
                for table in tables:
                    for row in table.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            item_name = cells[0].text.strip()
                            price_text = cells[1].text.strip()
                            data[item_name] = price_text
                
                raw_data.append(data)
                # Attente pour ne pas surcharger le serveur
                time.sleep(.1)
            except Exception as e:
                print(f"Erreur extraction coût de la vie {city}: {str(e)}")
        
        # Sauvegarde des données brutes
        self._save_raw_data(raw_data, 'cost_of_living')
        
        return raw_data
    
    def extract_health(self, cities):
        """
        Extraction des données de santé pour plusieurs villes
        
        Args:
            cities: liste de noms de villes
            
        Returns:
            raw_data: liste de dictionnaires contenant les données brutes
        """
        raw_data = []
        for city in cities:
            try:
                print(f"Extraction des données de santé pour {city}...")
                url = f"https://www.numbeo.com/health-care/in/{city}"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                data = {'City': city}
                
                # Extraction de l'index de santé général
                health_index = soup.find('td', string=re.compile('Health Care Index:'))
                if health_index and health_index.find_next_sibling():
                    data['Health Care Index'] = health_index.find_next_sibling().text.strip()
                
                # Extraction des composants de santé depuis la table
                table = soup.find('table', {'class': 'table_builder_with_value_explanation data_wide_table'})
                if table:
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            component = cols[0].text.strip()
                            value = cols[2].text.strip()
                            data[component] = value
                
                raw_data.append(data)
                time.sleep(.1)
                
            except Exception as e:
                print(f"Erreur extraction santé {city}: {str(e)}")
        
        # Sauvegarde des données brutes
        self._save_raw_data(raw_data, 'health')
        
        return raw_data
    
    def extract_crime(self, cities):
        """
        Extraction des données de criminalité pour plusieurs villes
        
        Args:
            cities: liste de noms de villes
            
        Returns:
            raw_data: liste de dictionnaires contenant les données brutes
        """
        raw_data = []
        for city in cities:
            try:
                print(f"Extraction des données de criminalité pour {city}...")
                url = f"https://www.numbeo.com/crime/in/{city}"
                response = requests.get(url, headers=self.headers)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                data = {'City': city}
                
                # Extraction des indices de criminalité
                crime_index = soup.find('td', string=re.compile('Crime Index:'))
                if crime_index and crime_index.find_next_sibling():
                    data['Crime Index'] = crime_index.find_next_sibling().text.strip()
                
                safety_index = soup.find('td', string=re.compile('Safety Index:'))
                if safety_index and safety_index.find_next_sibling():
                    data['Safety Index'] = safety_index.find_next_sibling().text.strip()
                
                # Extraction des composants de criminalité depuis la table
                table = soup.find('table', {'class': 'table_builder_with_value_explanation data_wide_table'})
                if table:
                    for row in table.find_all('tr')[1:]:  # Skip header row
                        cols = row.find_all('td')
                        if len(cols) >= 3:
                            component = cols[0].text.strip()
                            value = cols[2].text.strip()
                            data[component] = value
                
                raw_data.append(data)
                time.sleep(.1)
                
            except Exception as e:
                print(f"Erreur extraction criminalité {city}: {str(e)}")
        
        # Sauvegarde des données brutes
        self._save_raw_data(raw_data, 'crime')
        
        return raw_data
    
    def _save_raw_data(self, data, data_type):
        """
        Sauvegarde des données brutes dans des fichiers JSON et CSV
        
        Args:
            data: liste de dictionnaires de données
            data_type: type de données (cost_of_living, health, crime)
        """
        # Sauvegarde en JSON
        with open(f'data/extract/{data_type}_raw.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        
        # Sauvegarde en CSV
        df = pd.DataFrame(data)
        df.to_csv(f'data/extract/{data_type}_raw.csv', index=False)