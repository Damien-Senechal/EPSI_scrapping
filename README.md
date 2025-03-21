# Analyse des coûts et qualité de vie dans les villes françaises

## Description du projet
Ce projet extrait, nettoie, analyse et visualise des données sur le coût de la vie, la santé et la criminalité dans différentes villes françaises à partir du site Numbeo. Le projet implémente un pipeline complet d'extraction-transformation-chargement (ETL) et utilise des techniques d'apprentissage automatique pour analyser et prédire différentes métriques.

## Fonctionnalités principales
- **Extraction de données Web** : Scraping de données depuis Numbeo pour plusieurs villes françaises
- **Nettoyage et transformation** : Normalisation, gestion des valeurs manquantes et encodage
- **Analyse exploratoire** : Visualisations avancées des données
- **Apprentissage automatique** :
  - Modèles de régression pour prédire les prix immobiliers
  - Modèles de classification pour catégoriser les villes
  - Optimisation des hyperparamètres

## Structure du projet
```
project/
├── data/
│   ├── extract/            # Données brutes (JSON, CSV)
│   ├── transform/          # Données nettoyées
│   ├── load/               # Données finales pour analyse
│   └── models/             # Modèles ML sauvegardés
├── src/
│   ├── __init__.py
│   ├── scraping.py         # Extraction de données
│   ├── data_cleaning.py    # Transformation des données
│   ├── data_storage.py     # Persistance des données
│   ├── ml_analysis.py      # Modèles machine learning
│   ├── visualization.py    # Visualisation des données
│   └── utils.py            # Fonctions utilitaires
├── main.py                 # Script principal d'orchestration
├── requirements.txt        # Dépendances Python
└── README.md               # Documentation du projet
```

## Prérequis
- Python 3.12+
- pip (gestionnaire de paquets Python)
- Git

## Installation

```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/EPSI_scrapping.git
cd EPSI_scrapping

# Créer un environnement virtuel
python -m venv venv

# Activer l'environnement virtuel
# Sur Windows
venv\Scripts\activate
# Sur macOS/Linux
source venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### 1. Exécution du pipeline complet
Pour exécuter le processus complet de scraping, nettoyage et analyse ML :

```bash
python main.py
```

Ce script va :
1. Extraire les données de Numbeo pour les villes définies
2. Nettoyer et transformer les données
3. Effectuer les analyses par ML (régression et classification)
4. Générer les visualisations

### 2. Utilisation des modules individuels

Vous pouvez également utiliser les modules individuellement :

```python
# Extraction des données
from src.scraping import Numbeo
scraper = Numbeo()
cities = ["Paris", "Lyon", "Marseille"]
cost_data = scraper.extract_cost_of_living(cities)

# Nettoyage des données
from src.data_cleaning import DataCleaner
cleaner = DataCleaner()
cleaned_data = cleaner.clean_cost_of_living(cost_data)

# Analyse par ML
from src.ml_analysis import MLAnalysis
ml = MLAnalysis()
# ... utilisation des fonctions d'analyse ...
```

## Détails des données collectées

### 1. Coût de la vie
Les données de coût de la vie incluent:
- Prix des repas (restaurants, fast-food)
- Prix des boissons (café, bière, eau, etc.)
- Prix des produits alimentaires (pain, lait, viande, fruits, etc.)
- Coûts de transport (ticket de bus, essence, taxi)
- Coûts des logements (loyer, prix d'achat)
- Salaires moyens

### 2. Santé
Les données de santé incluent:
- Indices de qualité des soins médicaux
- Compétence du personnel médical
- Équipement pour diagnostics modernes
- Satisfaction concernant les coûts médicaux
- Temps d'attente dans les institutions médicales

### 3. Criminalité
Les données de criminalité incluent:
- Indices de criminalité et de sécurité
- Perception de la sécurité

## Modèles Machine Learning

### Modèles de régression
Le projet utilise plusieurs modèles de régression pour prédire les prix immobiliers:
- **Régression linéaire** : Modèle de base fournissant des prédictions interprétables
- **Random Forest Regression** : Modèle d'ensemble offrant généralement une meilleure performance
- **Support Vector Regression (SVR)** : Pour capturer des relations non-linéaires

Métriques d'évaluation: MSE, RMSE, coefficient de détermination (R²)

### Modèles de classification
Pour la classification des villes selon différentes métriques:
- **Random Forest Classifier** : Pour catégoriser les villes selon le niveau de vie
- **Support Vector Classification** : Alternative pour la classification

Métriques d'évaluation: précision (accuracy), score F1, matrice de confusion

## Dépannage
- **Erreurs de scraping** : Si Numbeo a modifié sa structure HTML, vous devrez peut-être ajuster les sélecteurs dans `scraping.py`
- **Erreurs de conversion des prix** : Vérifiez que les formats monétaires sont bien gérés dans `data_cleaning.py`
- **Problèmes de ML** : Si vous rencontrez des erreurs liées au ML, vérifiez que vos données d'entrée sont bien nettoyées et que toutes les valeurs manquantes sont gérées.

## Ressources
- [Documentation de BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Documentation de scikit-learn](https://scikit-learn.org/stable/documentation.html)
- [Documentation de Pandas](https://pandas.pydata.org/docs/)
- [Site Numbeo](https://www.numbeo.com/)