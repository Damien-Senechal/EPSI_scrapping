# src/scraping.py
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import json
import os
import logging
from urllib.parse import urljoin
from requests.exceptions import RequestException, HTTPError, ConnectionError, Timeout
from datetime import datetime
import random

# Configuration du logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='logs/scraping.log',
    filemode='a'
)
logger = logging.getLogger('numbeo_scraper')

class Numbeo:
    """
    Classe pour l'extraction des données depuis Numbeo.com
    Permet de récupérer des données sur le coût de la vie, la santé et la criminalité dans différentes villes.
    """
    
    def __init__(self, max_retries=3, retry_delay=5, timeout=30):
        """
        Initialise la classe de scraping avec les paramètres nécessaires
        
        Args:
            max_retries: nombre de tentatives en cas d'échec
            retry_delay: délai entre les tentatives (en secondes)
            timeout: délai d'expiration des requêtes (en secondes)
        """
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8'
        }
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout = timeout
        self.base_url = "https://www.numbeo.com/"
        
        # Création des répertoires de stockage
        os.makedirs('data/extract', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Journal du début de session
        logger.info(f"Initialisation de la session de scraping Numbeo ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    
    def _make_request(self, url):
        """
        Effectue une requête HTTP avec gestion des tentatives et des erreurs
        
        Args:
            url: URL à requêter
            
        Returns:
            response: objet de réponse requests ou None en cas d'échec
        """
        attempts = 0
        last_exception = None
        
        while attempts < self.max_retries:
            try:
                # Ajout d'un délai variable pour être moins identifiable comme bot
                delay = self.retry_delay + random.uniform(0.5, 2.0)
                if attempts > 0:
                    time.sleep(delay)
                
                logger.info(f"Tentative de requête vers {url} (tentative {attempts+1}/{self.max_retries})")
                response = requests.get(
                    url, 
                    headers=self.headers, 
                    timeout=self.timeout
                )
                response.raise_for_status()  # Lève une exception si le code HTTP n'est pas 2xx
                
                # Vérifier que le contenu est bien HTML (et pas une page de CAPTCHA)
                if "text/html" not in response.headers.get('Content-Type', ''):
                    raise ValueError(f"La réponse n'est pas HTML: {response.headers.get('Content-Type')}")
                
                # Vérifier que la page n'est pas une redirection ou une page d'erreur
                if "Access denied" in response.text or "CAPTCHA" in response.text:
                    raise ValueError("Accès détecté comme bot ou CAPTCHA demandé")
                
                return response
            
            except (RequestException, HTTPError, ConnectionError, Timeout, ValueError) as e:
                last_exception = e
                attempts += 1
                logger.warning(f"Échec de la requête {url}: {str(e)} - Tentative {attempts}/{self.max_retries}")
        
        # Si toutes les tentatives ont échoué
        logger.error(f"Échec définitif de la requête {url} après {self.max_retries} tentatives: {str(last_exception)}")
        return None
    
    def extract_cost_of_living(self, cities):
        """
        Extraction des données du coût de la vie pour plusieurs villes avec gestion d'erreurs améliorée
        
        Args:
            cities: liste de noms de villes
            
        Returns:
            raw_data: liste de dictionnaires contenant les données brutes
        """
        raw_data = []
        total_cities = len(cities)
        
        for idx, city in enumerate(cities):
            logger.info(f"Extraction des données de coût de la vie pour {city} ({idx+1}/{total_cities})")
            try:
                url = urljoin(self.base_url, f"cost-of-living/in/{city}?displayCurrency=EUR")
                response = self._make_request(url)
                
                if response is None:
                    logger.error(f"Impossible d'obtenir les données pour {city} - passage à la ville suivante")
                    # Ajout d'une entrée avec données minimales pour éviter des erreurs plus tard
                    raw_data.append({
                        'City': city,
                        'Extraction_Status': 'Failed',
                        'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Vérification de base pour s'assurer que nous sommes sur la bonne page
                title = soup.find('title')
                if title and city.lower() not in title.text.lower():
                    logger.warning(f"La page obtenue ne semble pas correspondre à {city} (titre: {title.text})")
                
                # Initialisation des données
                data = {
                    'City': city,
                    'Extraction_Status': 'Success',
                    'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Recherche des tableaux de prix
                tables = soup.find_all('table', class_='data_wide_table')
                
                if not tables:
                    logger.warning(f"Aucun tableau de données trouvé pour {city}")
                    # Fallback - chercher d'autres éléments
                    tables = soup.find_all('table')
                    if not tables:
                        logger.error(f"Aucun tableau trouvé pour {city} - structure HTML peut-être modifiée")
                        data['Extraction_Status'] = 'No_Tables_Found'
                        raw_data.append(data)
                        continue
                
                # Extraction des données des tableaux
                for table in tables:
                    rows = table.find_all('tr')
                    if not rows:
                        continue
                        
                    for row in rows:
                        cells = row.find_all('td')
                        if len(cells) >= 2:
                            try:
                                item_name = cells[0].text.strip()
                                price_text = cells[1].text.strip()
                                
                                if item_name and price_text:
                                    data[item_name] = price_text
                            except Exception as e:
                                logger.warning(f"Erreur lors de l'extraction d'une ligne pour {city}: {str(e)}")
                
                # Vérification que nous avons bien extrait des données
                item_count = len(data) - 3  # moins les 3 champs de métadonnées
                if item_count == 0:
                    logger.warning(f"Aucune donnée extraite pour {city} malgré des tableaux trouvés")
                    data['Extraction_Status'] = 'No_Data_Extracted'
                else:
                    logger.info(f"Extraction réussie pour {city}: {item_count} éléments extraits")
                
                raw_data.append(data)
                
                # Ajout d'un délai aléatoire pour éviter d'être détecté comme un bot
                delay = 1 + random.uniform(0.5, 2.0)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Erreur non gérée lors de l'extraction pour {city}: {str(e)}")
                # Ajout d'une entrée avec données minimales pour éviter des erreurs plus tard
                raw_data.append({
                    'City': city,
                    'Extraction_Status': 'Error',
                    'Extraction_Error': str(e),
                    'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sauvegarde des données brutes
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_raw_data(raw_data, f'cost_of_living_{timestamp}')
        
        # Filtrer les données qui ont échoué pour le retour
        success_data = [item for item in raw_data if item.get('Extraction_Status') == 'Success']
        logger.info(f"Extraction terminée: {len(success_data)}/{len(cities)} villes extraites avec succès")
        
        return raw_data  # Retourne toutes les données pour permettre un traitement des échecs
    
    def extract_health(self, cities):
        """
        Extraction des données de santé pour plusieurs villes avec gestion d'erreurs améliorée
        
        Args:
            cities: liste de noms de villes
            
        Returns:
            raw_data: liste de dictionnaires contenant les données brutes
        """
        raw_data = []
        total_cities = len(cities)
        
        for idx, city in enumerate(cities):
            logger.info(f"Extraction des données de santé pour {city} ({idx+1}/{total_cities})")
            try:
                url = urljoin(self.base_url, f"health-care/in/{city}")
                response = self._make_request(url)
                
                if response is None:
                    logger.error(f"Impossible d'obtenir les données de santé pour {city}")
                    raw_data.append({
                        'City': city,
                        'Extraction_Status': 'Failed',
                        'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Initialisation des données
                data = {
                    'City': city,
                    'Extraction_Status': 'Success',
                    'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Extraction de l'index de santé général
                health_index = None
                try:
                    # Recherche par texte exact
                    health_index = soup.find('td', string=re.compile('Health Care Index:'))
                    
                    # Si non trouvé, essayer des alternatives
                    if not health_index:
                        # Recherche par similarité de texte
                        health_labels = soup.find_all('td')
                        for label in health_labels:
                            if 'health' in label.text.lower() and 'index' in label.text.lower():
                                health_index = label
                                break
                except Exception as e:
                    logger.warning(f"Erreur lors de la recherche de l'index de santé pour {city}: {str(e)}")
                
                if health_index and health_index.find_next_sibling():
                    data['Health Care Index'] = health_index.find_next_sibling().text.strip()
                else:
                    logger.warning(f"Index de santé non trouvé pour {city}")
                
                # Extraction des composants de santé depuis la table
                try:
                    # Recherche spécifique
                    table = soup.find('table', {'class': 'table_builder_with_value_explanation data_wide_table'})
                    
                    # Si non trouvé, recherche plus générale
                    if not table:
                        tables = soup.find_all('table')
                        for t in tables:
                            if 'value_explanation' in str(t.get('class', [])):
                                table = t
                                break
                        
                    if table:
                        rows = table.find_all('tr')
                        
                        # Vérifier si c'est bien un tableau de données de santé
                        if rows and len(rows) > 1:
                            for row in rows[1:]:  # Skip header row
                                try:
                                    cols = row.find_all('td')
                                    if len(cols) >= 3:
                                        component = cols[0].text.strip()
                                        value = cols[2].text.strip()
                                        if component and value:
                                            data[component] = value
                                except Exception as e:
                                    logger.warning(f"Erreur lors de l'extraction d'une ligne de santé pour {city}: {str(e)}")
                    else:
                        logger.warning(f"Tableau des composants de santé non trouvé pour {city}")
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction du tableau de santé pour {city}: {str(e)}")
                
                # Vérification des données extraites
                item_count = len(data) - 3  # moins les champs de métadonnées
                if item_count == 0:
                    logger.warning(f"Aucune donnée de santé extraite pour {city}")
                    data['Extraction_Status'] = 'No_Data_Extracted'
                else:
                    logger.info(f"Extraction de santé réussie pour {city}: {item_count} éléments extraits")
                
                raw_data.append(data)
                
                # Délai aléatoire
                delay = 1 + random.uniform(0.5, 2.0)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Erreur non gérée lors de l'extraction des données de santé pour {city}: {str(e)}")
                raw_data.append({
                    'City': city,
                    'Extraction_Status': 'Error',
                    'Extraction_Error': str(e),
                    'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sauvegarde des données brutes
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_raw_data(raw_data, f'health_{timestamp}')
        
        return raw_data
    
    def extract_crime(self, cities):
        """
        Extraction des données de criminalité pour plusieurs villes avec gestion d'erreurs améliorée
        
        Args:
            cities: liste de noms de villes
            
        Returns:
            raw_data: liste de dictionnaires contenant les données brutes
        """
        raw_data = []
        total_cities = len(cities)
        
        for idx, city in enumerate(cities):
            logger.info(f"Extraction des données de criminalité pour {city} ({idx+1}/{total_cities})")
            try:
                url = urljoin(self.base_url, f"crime/in/{city}")
                response = self._make_request(url)
                
                if response is None:
                    logger.error(f"Impossible d'obtenir les données de criminalité pour {city}")
                    raw_data.append({
                        'City': city,
                        'Extraction_Status': 'Failed',
                        'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    })
                    continue
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Initialisation des données
                data = {
                    'City': city,
                    'Extraction_Status': 'Success',
                    'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Extraction des indices de criminalité et sécurité
                crime_index = safety_index = None
                
                try:
                    # Recherche par texte exact
                    crime_index = soup.find('td', string=re.compile('Crime Index:'))
                    safety_index = soup.find('td', string=re.compile('Safety Index:'))
                    
                    # Si non trouvés, essayer des approches alternatives
                    if not crime_index or not safety_index:
                        indices = soup.find_all('td')
                        for idx in indices:
                            text = idx.text.lower()
                            if 'crime index' in text and not crime_index:
                                crime_index = idx
                            elif 'safety index' in text and not safety_index:
                                safety_index = idx
                except Exception as e:
                    logger.warning(f"Erreur lors de la recherche des indices pour {city}: {str(e)}")
                
                if crime_index and crime_index.find_next_sibling():
                    data['Crime Index'] = crime_index.find_next_sibling().text.strip()
                
                if safety_index and safety_index.find_next_sibling():
                    data['Safety Index'] = safety_index.find_next_sibling().text.strip()
                
                # Extraction des composants de criminalité depuis la table
                try:
                    table = soup.find('table', {'class': 'table_builder_with_value_explanation data_wide_table'})
                    
                    # Si non trouvé, recherche plus générale
                    if not table:
                        tables = soup.find_all('table')
                        for t in tables:
                            if 'value_explanation' in str(t.get('class', [])):
                                table = t
                                break
                    
                    if table:
                        rows = table.find_all('tr')
                        
                        if rows and len(rows) > 1:
                            for row in rows[1:]:  # Skip header row
                                try:
                                    cols = row.find_all('td')
                                    if len(cols) >= 3:
                                        component = cols[0].text.strip()
                                        value = cols[2].text.strip()
                                        if component and value:
                                            data[component] = value
                                except Exception as e:
                                    logger.warning(f"Erreur lors de l'extraction d'une ligne de criminalité pour {city}: {str(e)}")
                    else:
                        logger.warning(f"Tableau des composants de criminalité non trouvé pour {city}")
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction du tableau de criminalité pour {city}: {str(e)}")
                
                # Vérification des données extraites
                item_count = len(data) - 3  # moins les champs de métadonnées
                if item_count == 0:
                    logger.warning(f"Aucune donnée de criminalité extraite pour {city}")
                    data['Extraction_Status'] = 'No_Data_Extracted'
                else:
                    logger.info(f"Extraction de criminalité réussie pour {city}: {item_count} éléments extraits")
                
                raw_data.append(data)
                
                # Délai aléatoire
                delay = 1 + random.uniform(0.5, 2.0)
                time.sleep(delay)
                
            except Exception as e:
                logger.error(f"Erreur non gérée lors de l'extraction des données de criminalité pour {city}: {str(e)}")
                raw_data.append({
                    'City': city,
                    'Extraction_Status': 'Error',
                    'Extraction_Error': str(e),
                    'Extraction_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                })
        
        # Sauvegarde des données brutes
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_raw_data(raw_data, f'crime_{timestamp}')
        
        return raw_data
    
    def _save_raw_data(self, data, data_type):
        """
        Sauvegarde des données brutes dans des fichiers JSON et CSV avec horodatage
        
        Args:
            data: liste de dictionnaires de données
            data_type: type de données (cost_of_living, health, crime)
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Création du répertoire d'extraction si nécessaire
        os.makedirs('data/extract', exist_ok=True)
        
        # Sauvegarde en JSON
        json_filename = f'data/extract/{data_type}_raw.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        logger.info(f"Données brutes sauvegardées en JSON: {json_filename}")
        
        # Sauvegarde en CSV
        try:
            df = pd.DataFrame(data)
            csv_filename = f'data/extract/{data_type}_raw.csv'
            df.to_csv(csv_filename, index=False)
            logger.info(f"Données brutes sauvegardées en CSV: {csv_filename}")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde en CSV: {str(e)}")
        
        # Sauvegarde d'une copie horodatée pour l'historique
        try:
            with open(f'data/extract/{data_type}_raw_{timestamp}.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logger.info(f"Copie horodatée sauvegardée: {data_type}_raw_{timestamp}.json")
        except Exception as e:
            logger.error(f"Erreur lors de la sauvegarde de la copie horodatée: {str(e)}")
    
    def verify_data_completeness(self, data_list, required_fields=None, min_required=1):
        """
        Vérifie la complétude des données extraites et tente des réparations si nécessaire
        
        Args:
            data_list: liste de dictionnaires de données à vérifier
            required_fields: liste des champs obligatoires à vérifier
            min_required: nombre minimal de champs requis pour qu'une entrée soit considérée comme valide
            
        Returns:
            verified_data: liste des données vérifiées, réparées si possible
        """
        if not data_list:
            logger.error("Aucune donnée à vérifier")
            return []
        
        # Si aucun champ requis n'est spécifié, utilisons la 'City' comme minimum
        if required_fields is None:
            required_fields = ['City']
        
        verified_data = []
        cities_with_issues = []
        
        for item in data_list:
            city = item.get('City', 'Unknown')
            
            # Vérification des champs requis
            missing_fields = [field for field in required_fields if field not in item]
            
            if len(missing_fields) > 0:
                logger.warning(f"Données incomplètes pour {city}: champs manquants {missing_fields}")
                cities_with_issues.append(city)
                
                # Tenter une réparation - ajouter les champs manquants avec valeurs vides
                for field in missing_fields:
                    item[field] = None
            
            # Vérification du nombre minimal de champs (hors métadonnées)
            data_fields = [field for field in item.keys() if field not in ['City', 'Extraction_Status', 'Extraction_Time', 'Extraction_Error']]
            
            if len(data_fields) < min_required:
                logger.warning(f"Trop peu de données pour {city}: {len(data_fields)} champs < {min_required} requis")
                cities_with_issues.append(city)
                
                # Marquer comme donnée problématique mais conserver
                item['Extraction_Status'] = 'Insufficient_Data'
            
            verified_data.append(item)
        
        # Rapport final
        if cities_with_issues:
            logger.warning(f"{len(cities_with_issues)}/{len(data_list)} villes avec des problèmes de données: {', '.join(set(cities_with_issues))}")
        else:
            logger.info(f"Vérification réussie pour toutes les {len(data_list)} villes")
        
        return verified_data