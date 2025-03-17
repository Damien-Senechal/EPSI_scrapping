# src/utils.py
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import logging

def setup_logger(log_file='app.log'):
    """
    Configure un logger pour le suivi des opérations
    
    Args:
        log_file: chemin du fichier de log
        
    Returns:
        logger: objet logger configuré
    """
    # Création du répertoire de logs si nécessaire
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configuration du logger
    logger = logging.getLogger('city_data_analysis')
    logger.setLevel(logging.INFO)
    
    # Handler de fichier pour enregistrer les logs
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Handler de console pour afficher les logs
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Format des logs
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Ajout des handlers au logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def create_directory_structure():
    """
    Crée la structure de répertoires nécessaire pour le projet
    """
    directories = [
        'data/extract',
        'data/transform',
        'data/load',
        'data/models',
        'data/visualizations',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Structure de répertoires créée avec succès.")

def save_figure(fig, filename, directory='data/visualizations', format='png', dpi=300):
    """
    Sauvegarde une figure matplotlib avec gestion automatique des répertoires
    
    Args:
        fig: figure matplotlib à sauvegarder
        filename: nom du fichier sans extension
        directory: répertoire de sauvegarde
        format: format de l'image (png, pdf, svg, etc.)
        dpi: résolution de l'image (dots per inch)
    """
    # Création du répertoire si nécessaire
    os.makedirs(directory, exist_ok=True)
    
    # Chemin complet du fichier
    filepath = os.path.join(directory, f"{filename}.{format}")
    
    # Sauvegarde
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Figure sauvegardée: {filepath}")

def save_plotly_figure(fig, filename, directory='data/visualizations', format='html'):
    """
    Sauvegarde une figure plotly avec gestion automatique des répertoires
    
    Args:
        fig: figure plotly à sauvegarder
        filename: nom du fichier sans extension
        directory: répertoire de sauvegarde
        format: format (html, png, jpg, pdf, svg)
    """
    # Création du répertoire si nécessaire
    os.makedirs(directory, exist_ok=True)
    
    # Chemin complet du fichier
    filepath = os.path.join(directory, f"{filename}.{format}")
    
    # Sauvegarde selon le format
    if format == 'html':
        fig.write_html(filepath)
    else:
        fig.write_image(filepath)
    
    print(f"Figure Plotly sauvegardée: {filepath}")

def export_results(results, filename, directory='data/results'):
    """
    Exporte les résultats d'analyse dans un fichier JSON
    
    Args:
        results: dictionnaire de résultats
        filename: nom du fichier sans extension
        directory: répertoire de sauvegarde
    """
    # Création du répertoire si nécessaire
    os.makedirs(directory, exist_ok=True)
    
    # Chemin complet du fichier
    filepath = os.path.join(directory, f"{filename}.json")
    
    # Conversion des objets non sérialisables
    serializable_results = {}
    for key, value in results.items():
        if hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
            # Conversion des arrays numpy en listes
            serializable_results[key] = value.tolist()
        elif hasattr(value, '__dict__'):
            # Pour les objets avec des attributs
            serializable_results[key] = str(value)
        else:
            # Pour les types simples
            try:
                json.dumps({key: value})
                serializable_results[key] = value
            except (TypeError, OverflowError):
                serializable_results[key] = str(value)
    
    # Sauvegarde en JSON
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=4)
    
    print(f"Résultats exportés: {filepath}")

def create_interactive_map(df, map_title='Carte des villes', value_column=None, size_column=None):
    """
    Crée une carte interactive avec Plotly pour visualiser les villes
    
    Args:
        df: DataFrame avec une colonne 'City' et éventuellement des colonnes de valeurs
        map_title: titre de la carte
        value_column: colonne à utiliser pour la couleur des points
        size_column: colonne à utiliser pour la taille des points
        
    Returns:
        fig: figure plotly de la carte
    """
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    
    # Dictionnaire des coordonnées géographiques des villes françaises
    city_coords = {
        'Paris': (48.8566, 2.3522),
        'Lyon': (45.7578, 4.8320),
        'Marseille': (43.2965, 5.3698),
        'Toulouse': (43.6047, 1.4442),
        'Bordeaux': (44.8378, -0.5792),
        'Nantes': (47.2184, -1.5536),
        'Nice': (43.7102, 7.2620),
        'Strasbourg': (48.5734, 7.7521),
        'Montpellier': (43.6108, 3.8767),
        'Lille': (50.6292, 3.0573),
        'Rennes': (48.1173, -1.6778),
        'Reims': (49.2583, 4.0317),
        'Le Havre': (49.4944, 0.1079),
        'Toulon': (43.1242, 5.9280),
        'Grenoble': (45.1885, 5.7245),
        'Dijon': (47.3220, 5.0415),
        'Angers': (47.4784, -0.5632),
        'Saint-Étienne': (45.4397, 4.3872),
        'Nîmes': (43.8367, 4.3601),
        'Villeurbanne': (45.7679, 4.8830)
    }
    
    # Convert df to a DataFrame if it's not already one
    if not isinstance(df, pd.DataFrame):
        print("Avertissement: df n'est pas un DataFrame.")
        # Create a fallback figure with just the cities
        fallback_fig = go.Figure()
        fallback_fig.add_trace(go.Scattermapbox(
            lat=[coords[0] for coords in city_coords.values()],
            lon=[coords[1] for coords in city_coords.values()],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=list(city_coords.keys()),
            hoverinfo='text'
        ))
        fallback_fig.update_layout(
            title=map_title,
            mapbox=dict(
                style="open-street-map",
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334}  # Center of France
            ),
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        return fallback_fig
    
    # Make a copy to avoid modifying the original
    df_copy = df.copy()
    
    # If DataFrame is empty, return a fallback map
    if df.empty:
        print("Avertissement: DataFrame vide.")
        fallback_fig = go.Figure()
        fallback_fig.add_trace(go.Scattermapbox(
            lat=[coords[0] for coords in city_coords.values()],
            lon=[coords[1] for coords in city_coords.values()],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=list(city_coords.keys()),
            hoverinfo='text'
        ))
        fallback_fig.update_layout(
            title=map_title,
            mapbox=dict(
                style="open-street-map",
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334}
            ),
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        return fallback_fig
    
    # Préparation des données pour la carte
    map_data = []
    
    # Check if the index is named 'City' or contains city names
    if df.index.name == 'City':
        city_list = df.index.tolist()
        for city in city_list:
            if city in city_coords:
                lat, lon = city_coords[city]
                data_point = {
                    'City': city,
                    'Latitude': lat,
                    'Longitude': lon
                }
                # Add other columns if they exist
                if value_column and value_column in df.columns:
                    data_point[value_column] = df.loc[city, value_column]
                if size_column and size_column in df.columns:
                    data_point[size_column] = df.loc[city, size_column]
                map_data.append(data_point)
    
    # Check if 'City' is in the columns
    elif 'City' in df.columns:
        for _, row in df.iterrows():
            city = row['City']
            if city in city_coords:
                lat, lon = city_coords[city]
                data_point = {
                    'City': city,
                    'Latitude': lat,
                    'Longitude': lon
                }
                # Add other columns if they exist
                if value_column and value_column in df.columns:
                    data_point[value_column] = row[value_column]
                if size_column and size_column in df.columns:
                    data_point[size_column] = row[size_column]
                map_data.append(data_point)
    
    # If index contains city names directly
    else:
        # Reset index to get it as a column
        df_with_index = df.reset_index()
        
        # Assume the first column contains city names
        index_col = df_with_index.columns[0]
        
        for _, row in df_with_index.iterrows():
            # Get the city name from the first column
            possible_city = str(row[index_col])
            
            # Check if this is a known city
            city_found = False
            for city in city_coords:
                if city.lower() in possible_city.lower():
                    lat, lon = city_coords[city]
                    data_point = {
                        'City': city,
                        'Latitude': lat,
                        'Longitude': lon
                    }
                    # Add other columns if they exist
                    if value_column and value_column in df.columns:
                        data_point[value_column] = row[value_column] if value_column in row else None
                    if size_column and size_column in df.columns:
                        data_point[size_column] = row[size_column] if size_column in row else None
                    map_data.append(data_point)
                    city_found = True
                    break
            
            # If it's a numeric index, just use direct mapping
            if not city_found and index_col == 'index' and isinstance(row[index_col], int):
                idx = row[index_col]
                if idx < len(city_coords):
                    city = list(city_coords.keys())[idx]
                    lat, lon = city_coords[city]
                    data_point = {
                        'City': city,
                        'Latitude': lat,
                        'Longitude': lon
                    }
                    # Add other columns if they exist
                    if value_column and value_column in df.columns:
                        data_point[value_column] = row[value_column] if value_column in row else None
                    if size_column and size_column in df.columns:
                        data_point[size_column] = row[size_column] if size_column in row else None
                    map_data.append(data_point)
    
    # If we couldn't find any valid cities, return a fallback map
    if not map_data:
        print("Avertissement: Aucune donnée valide pour la carte.")
        fallback_fig = go.Figure()
        fallback_fig.add_trace(go.Scattermapbox(
            lat=[coords[0] for coords in city_coords.values()],
            lon=[coords[1] for coords in city_coords.values()],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=list(city_coords.keys()),
            hoverinfo='text'
        ))
        fallback_fig.update_layout(
            title=map_title,
            mapbox=dict(
                style="open-street-map",
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334}
            ),
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        return fallback_fig
    
    # Convert list of dictionaries to DataFrame
    map_df = pd.DataFrame(map_data)
    
    # Create the map
    try:
        if value_column and value_column in map_df.columns and size_column and size_column in map_df.columns:
            fig = px.scatter_mapbox(
                map_df,
                lat='Latitude',
                lon='Longitude',
                color=value_column,
                size=size_column,
                hover_name='City',
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334},
                mapbox_style="open-street-map",
                title=map_title
            )
        elif value_column and value_column in map_df.columns:
            fig = px.scatter_mapbox(
                map_df,
                lat='Latitude',
                lon='Longitude',
                color=value_column,
                hover_name='City',
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334},
                mapbox_style="open-street-map",
                title=map_title
            )
        else:
            fig = px.scatter_mapbox(
                map_df,
                lat='Latitude',
                lon='Longitude',
                hover_name='City',
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334},
                mapbox_style="open-street-map",
                title=map_title
            )
        
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            height=600
        )
        
        return fig
    
    except Exception as e:
        print(f"Erreur lors de la création de la carte: {e}")
        # Create a fallback figure if there's an error
        fallback_fig = go.Figure()
        fallback_fig.add_trace(go.Scattermapbox(
            lat=[row['Latitude'] for row in map_data],
            lon=[row['Longitude'] for row in map_data],
            mode='markers',
            marker=dict(size=10, color='blue'),
            text=[row['City'] for row in map_data],
            hoverinfo='text'
        ))
        fallback_fig.update_layout(
            title=map_title,
            mapbox=dict(
                style="open-street-map",
                zoom=5,
                center={"lat": 46.603354, "lon": 1.888334}
            ),
            height=600,
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        return fallback_fig

def compare_dataframes(df1, df2, index_col='City'):
    """
    Compare deux DataFrames et met en évidence les différences
    
    Args:
        df1: premier DataFrame
        df2: deuxième DataFrame
        index_col: colonne à utiliser comme index pour la comparaison
        
    Returns:
        diff_df: DataFrame avec les différences
    """
    # Définir l'index pour la comparaison
    if index_col in df1.columns:
        df1 = df1.set_index(index_col)
    if index_col in df2.columns:
        df2 = df2.set_index(index_col)
    
    # Identifier les colonnes communes
    common_columns = list(set(df1.columns) & set(df2.columns))
    
    # Création d'un DataFrame pour les différences
    diff_data = []
    
    for idx in set(list(df1.index) + list(df2.index)):
        row = {'Index': idx}
        
        # Vérifier si l'index existe dans les deux DataFrames
        if idx in df1.index and idx in df2.index:
            for col in common_columns:
                val1 = df1.loc[idx, col]
                val2 = df2.loc[idx, col]
                
                if val1 != val2:
                    row[f"{col}_df1"] = val1
                    row[f"{col}_df2"] = val2
                    row[f"{col}_diff"] = val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else "N/A"
        else:
            row['Status'] = 'Only in df1' if idx in df1.index else 'Only in df2'
        
        if len(row) > 1:  # Si des différences ont été trouvées
            diff_data.append(row)
    
    # Création du DataFrame de différences
    diff_df = pd.DataFrame(diff_data)
    
    return diff_df

def generate_timestamp():
    """
    Génère un horodatage formaté pour les noms de fichiers
    
    Returns:
        timestamp: chaîne d'horodatage formatée
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")