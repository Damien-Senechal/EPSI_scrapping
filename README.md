# Analyse des coûts et qualité de vie dans les villes françaises

## Description
Ce projet extrait, nettoie, analyse et visualise des données sur le coût de la vie, la santé et la criminalité dans différentes villes françaises à partir du site Numbeo. Le projet implémente un pipeline complet d'extraction-transformation-chargement (ETL) et utilise des techniques d'apprentissage automatique pour analyser et prédire différentes métriques.

## Caractéristiques
- **Extraction de données Web** : Scraping de données depuis Numbeo pour plusieurs villes françaises
- **Nettoyage et transformation** : Normalisation, gestion des valeurs manquantes et encodage
- **Analyse exploratoire** : Visualisations avancées des données
- **Apprentissage automatique** :
  - Modèles de régression pour prédire les prix immobiliers
  - Modèles de classification pour catégoriser les villes
  - Optimisation des hyperparamètres
- **Stockage de données** : Sauvegarde en fichiers CSV/JSON et option de base de données SQLite
- **Visualisation interactive** : Graphiques statiques et interactifs
- **Conteneurisation** : Support Docker pour déploiement facile

## Structure du projet
```
project/
├── .devcontainer/          # Configuration pour développement en conteneur
├── airflow/                # DAGs Airflow pour orchestration 
├── data/
│   ├── extract/            # Données brutes
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
├── notebooks/              # Pour exploration et documentation
├── main.py                 # Script principal d'orchestration
├── Dockerfile              # Pour conteneurisation
├── requirements.txt        # Dépendances Python
└── README.md               # Documentation du projet
```

## Prérequis
- Python 3.12+
- pip (gestionnaire de paquets Python)
- Git
- Docker (optionnel)

## Installation

### Méthode 1: Installation standard
```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/analyse-cout-vie.git
cd analyse-cout-vie

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

### Méthode 2: Utilisation de Docker
```bash
# Cloner le dépôt
git clone https://github.com/votre-nom/analyse-cout-vie.git
cd analyse-cout-vie

# Construire l'image Docker
docker build -t analyse-cout-vie .

# Exécuter le conteneur
docker run -it analyse-cout-vie
```

### Méthode 3: Utilisation de Dev Containers
Si vous utilisez VS Code avec l'extension Dev Containers, vous pouvez simplement ouvrir le dossier du projet et VS Code vous proposera d'ouvrir le projet dans un conteneur.

## Utilisation

### Exécution du pipeline complet
```bash
python main.py
```

### Modules individuels
Vous pouvez également utiliser les modules individuellement pour des analyses spécifiques:

```python
from src.scraping import Numbeo
from src.data_cleaning import DataCleaner
from src.ml_analysis import MLAnalysis
from src.visualization import Visualization

# Extraction des données
numbeo = Numbeo()
cities = ["Paris", "Lyon", "Marseille"]
cost_data = numbeo.extract_cost_of_living(cities)

# Nettoyage des données
cleaner = DataCleaner()
cleaned_data = cleaner.clean_cost_of_living(cost_data)

# Analyse par apprentissage automatique
ml = MLAnalysis()
# ... utilisation des fonctions d'analyse ...

# Visualisations
viz = Visualization()
# ... création de visualisations ...
```

## Exemples d'analyses

### Prédiction des prix immobiliers
Le projet inclut des modèles de régression pour prédire les prix des logements en fonction d'autres indicateurs économiques et sociaux:
- Régression linéaire simple
- Random Forest Regression
- Support Vector Regression (SVR)

### Classification des villes
Classification des villes selon différentes métriques:
- Niveau de vie (basé sur le salaire moyen)
- Rapport qualité-prix (qualité de vie vs coût)
- Attractivité globale

## Extension et personnalisation

### Ajout de nouvelles villes
Modifiez la liste des villes dans `main.py` pour inclure d'autres villes françaises ou internationales:

```python
cities = ["Paris", "Lyon", "Marseille", "Toulouse", "Bordeaux", "Nantes", "Nice", "Strasbourg", "Montpellier"]
```

### Ajout de nouvelles sources de données
Pour ajouter de nouvelles sources de données:
1. Créez une nouvelle méthode d'extraction dans la classe `Numbeo`
2. Ajoutez une méthode de nettoyage correspondante dans `DataCleaner`
3. Mettez à jour la fonction `save_final_data` dans `DataStorage` pour inclure les nouvelles données
4. Utilisez les données dans vos analyses et visualisations

## Contribution
Les contributions sont les bienvenues! N'hésitez pas à:
- Signaler des bugs
- Ajouter de nouvelles fonctionnalités
- Améliorer la documentation
- Proposer des corrections ou optimisations

## Licence
Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.